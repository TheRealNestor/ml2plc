import re
from pathlib import Path
from typing import Tuple, List

# Update import path based on your project structure
from .python_builder import PyCodeBuilder


def translate_st_to_python(st_code: str) -> Tuple[str, str]:
    """
    Translate a subset of Structured Text (ST) code to Python.

    Args:
        st_code: The Structured Text code to be translated.
    Returns:
        A tuple containing the translated Python code and the function name.
    """
    # Extract the FUNCTION_BLOCK name
    function_name_match = re.search(r"FUNCTION_BLOCK (\w+)", st_code)
    function_name = (
        function_name_match.group(1) if function_name_match else "translated_function"
    )

    # Remove FUNCTION_BLOCK and END_FUNCTION_BLOCK
    st_code = re.sub(r"FUNCTION_BLOCK (\w+)", r"", st_code)
    st_code = st_code.replace("END_FUNCTION_BLOCK", "")

    # Extract variable blocks
    variables: List[Tuple[str, str]] = []  # (block_content, var_type)

    def collect_variables(match, var_type):
        variables.append((match.group(1), var_type))
        return ""

    # Extract VAR_INPUT, VAR_OUTPUT, VAR CONSTANT, and VAR blocks
    st_code = re.sub(
        r"VAR_INPUT(.*?)END_VAR",
        lambda m: collect_variables(m, "input"),
        st_code,
        flags=re.DOTALL,
    )
    st_code = re.sub(
        r"VAR_OUTPUT(.*?)END_VAR",
        lambda m: collect_variables(m, "output"),
        st_code,
        flags=re.DOTALL,
    )
    st_code = re.sub(
        r"VAR CONSTANT(.*?)END_VAR",
        lambda m: collect_variables(m, "constant"),
        st_code,
        flags=re.DOTALL,
    )
    st_code = re.sub(
        r"VAR(.*?)END_VAR",
        lambda m: collect_variables(m, "var"),
        st_code,
        flags=re.DOTALL,
    )

    # Build the Python code
    builder = PyCodeBuilder()
    builder.add_import("numpy", ["exp"])

    with builder.function(function_name, ["input_data"]):
        # Add variable declarations
        # First, add non-input variable declarations and collect input decls
        input_decls: dict = {}
        for block_content, var_type in variables:
            if var_type == "input":
                # Parse input declarations for later normalization decisions
                input_decls.update(parse_var_input_declarations(block_content))
            else:
                add_variables(builder, block_content, var_type)

        # Normalize runtime input_data to scalar ONLY when the ST declaration
        # declares a scalar (not an ARRAY). This preserves array semantics
        # when the ST explicitly declares an ARRAY, and is future-proof.
        builder.add_line()
        builder.add_line(
            "# Normalize runtime input to scalar only when ST declares a scalar input"
        )
        input_decl = input_decls.get("input_data")
        if input_decl is None:
            # No input declaration found — be conservative: do not normalize.
            builder.add_line(
                "# No VAR_INPUT declaration found for input_data — preserve runtime shape"
            )
        else:
            # Heuristic: if the ST logic indexes into `input_data` anywhere, treat
            # the runtime input as an array even if the VAR_INPUT declaration
            # (erroneously) marks it as scalar. This avoids collapsing to a
            # Python scalar and then having later code attempt to index it,
            # which produces "invalid index to scalar variable" errors.
            # This is conservative and non-destructive: it only changes the
            # translator's runtime normalization behavior, not the original ST.
            uses_indexing = bool(__import__("re").search(r"\binput_data\s*\[", st_code))
            if not input_decl.get("is_array", False) and uses_indexing:
                # Treat as array at runtime
                builder.add_line(
                    "# Detected array-style access to input_data in ST; preserve as array at runtime"
                )
            elif not input_decl.get("is_array", False):
                # Scalar declaration and no array access detected: collapse singleton
                builder.add_line(
                    "if hasattr(input_data, '__len__') and len(input_data) == 1:"
                )
                with builder.indent():
                    builder.add_line("input_data = input_data[0]")
            else:
                builder.add_line(
                    "# input_data declared as ARRAY in ST — do not collapse to scalar"
                )

        # Parse and add the logic
        parse_st_logic(builder, st_code)

        builder.add_line()
        builder.add_line("return output_data")

    python_code = builder.build().to_string()
    return python_code, function_name


def add_variables(builder: PyCodeBuilder, variables_block: str, var_type: str):
    """Add variable declarations to the builder."""
    if var_type == "input":
        # Input variables are function parameters and are handled separately
        return

    for line in variables_block.splitlines():
        line = line.strip()
        if not line:
            continue

        # Array with initialization
        match = re.match(r"(\w+) : ARRAY\[\d+\.\.(\d+)\] OF (\w+) := (.+);", line)
        if match:
            name, size, dtype, values = match.groups()
            builder.add_line(f"{name} = {values}  # {var_type} variable")
            continue

        # Array without initialization
        match = re.match(r"(\w+) : ARRAY\[\d+\.\.(\d+)\] OF (\w+);", line)
        if match:
            name, size, dtype = match.groups()
            size = int(size) + 1
            default_value = "0.0" if dtype == "REAL" else "0"
            builder.add_line(
                f"{name} = [{default_value}] * {size}  # {var_type} variable"
            )
            continue

        # Scalar with initialization
        match = re.match(r"(\w+) : (\w+) := (.+);", line)
        if match:
            name, dtype, value = match.groups()
            builder.add_line(f"{name} = {value}  # {var_type} variable")
            continue

        # Scalar without initialization
        match = re.match(r"(\w+) : (\w+);", line)
        if match:
            name, dtype = match.groups()
            default_value = "0.0" if dtype == "REAL" else "0"
            builder.add_line(f"{name} = {default_value}  # {var_type} variable")
            continue


def parse_st_logic(builder: PyCodeBuilder, st_code: str):
    """Parse ST logic and add to builder with proper indentation."""
    # Preprocess: normalize line endings and split
    lines = st_code.strip().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Handle comments
        if line.startswith("//"):
            builder.add_line(f"# {line[2:].strip()}")
            continue

        if line.startswith("(*"):
            comment = line.replace("(*", "").replace("*)", "").strip()
            builder.add_line(f"# {comment}")
            continue

        # Handle FOR loops
        for_match = re.match(r"FOR (\w+) := (\d+) TO (\d+) DO", line)
        if for_match:
            var, start, end = for_match.groups()
            # Collect loop body
            loop_body, i = collect_block(lines, i, "END_FOR")
            with builder.for_loop(var, int(start), int(end)):
                parse_st_logic(builder, "\n".join(loop_body))
            continue

        # Handle IF statements
        if_match = re.match(r"IF (.+) THEN", line)
        if if_match:
            condition = translate_expression(if_match.group(1))
            # Collect if body
            if_body, i = collect_block(lines, i, "END_IF")
            with builder.if_block(condition):
                parse_st_logic(builder, "\n".join(if_body))
            continue

        # Handle assignments
        assign_match = re.match(r"(.+) := (.+);", line)
        if assign_match:
            lhs = translate_expression(assign_match.group(1))
            rhs = translate_expression(assign_match.group(2))
            builder.add_line(f"{lhs} = {rhs}")
            continue

        # Skip END markers (should be handled by collect_block)
        if line.startswith("END_"):
            continue


def collect_block(
    lines: List[str], start_idx: int, end_marker: str
) -> Tuple[List[str], int]:
    """Collect lines until the end marker, handling nested blocks."""
    body = []
    depth = 1
    i = start_idx

    while i < len(lines) and depth > 0:
        line = lines[i].strip()

        # Track nesting
        if re.match(r"FOR .+ DO", line) or re.match(r"IF .+ THEN", line):
            depth += 1
        elif line.startswith("END_FOR") or line.startswith("END_IF"):
            depth -= 1
            if depth == 0:
                i += 1
                break

        body.append(lines[i])
        i += 1

    return body, i


def translate_expression(expr: str) -> str:
    """Translate an ST expression to Python."""
    # Replace array access
    expr = re.sub(r"(\w+)\[(.+?)\]", r"\1[\2]", expr)

    # Replace functions
    expr = re.sub(r"MAX\((.+?),\s*(.+?)\)", r"max(\1, \2)", expr)
    expr = re.sub(r"EXP\((.+?)\)", r"exp(\1)", expr)

    # Replace type conversion functions (identity functions in Python)
    expr = re.sub(r"DINT_TO_LINT\((.+?)\)", r"\1", expr)
    expr = re.sub(r"LINT_TO_DINT\((.+?)\)", r"\1", expr)

    # Replace operators
    expr = expr.replace(":=", "=")

    # Replace logical operators (ST → Python)
    expr = re.sub(r"\bAND\b", "and", expr)
    expr = re.sub(r"\bOR\b", "or", expr)
    expr = re.sub(r"\bNOT\b", "not", expr)

    # Remove trailing semicolon
    expr = expr.rstrip(";")

    return expr


def parse_var_input_declarations(block: str) -> dict:
    """Parse VAR_INPUT block content and return a mapping of variable name -> info.

    Info dict contains:
      - is_array: bool
      - array_dims: list[int] | None
    """
    decls: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match ARRAY declarations: e.g. name : ARRAY[0..19] OF REAL;
        m = re.match(r"(\w+)\s*:\s*ARRAY\[(.*?)\]\s*OF\s*(\w+)", line)
        if m:
            name = m.group(1)
            dims = m.group(2)
            # dims could be '0..N' or '0..N, 0..M'
            parts = [p.strip() for p in dims.split(",")]
            dims_ints: list[int] = []
            for p in parts:
                mm = re.match(r"0\.\.(\d+)", p)
                if mm:
                    dims_ints.append(int(mm.group(1)) + 1)
                else:
                    # Couldn't parse bounds; treat as array with unknown size
                    dims_ints.append(0)

            decls[name] = {
                "is_array": True,
                "array_dims": dims_ints,
            }
            continue

        # Match scalar declaration: name : REAL;
        m2 = re.match(r"(\w+)\s*:\s*(\w+)\s*(?:;|:=)", line)
        if m2:
            name = m2.group(1)
            decls[name] = {"is_array": False, "array_dims": None}

    return decls


if __name__ == "__main__":
    st_folder = Path("examples/models/structured_text")
    st_file = st_folder / "conv_temp.st"

    with open(st_file, "r") as file:
        st_code = file.read()

    save_folder = Path("src/translation_validation/tmp")
    save_file = save_folder / "test.py"

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    python_code, function_name = translate_st_to_python(st_code)

    with open(save_file, "w") as file:
        file.write(python_code)

    print(f"Translated {function_name} to {save_file}")
