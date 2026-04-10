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
        for block_content, var_type in variables:
            add_variables(builder, block_content, var_type)

        builder.add_line()

        # Parse and add the logic
        parse_st_logic(builder, st_code)

        builder.add_line()
        builder.add_line("return output_data")

    python_code = builder.build().to_string()
    return python_code, function_name


def add_variables(builder: PyCodeBuilder, variables_block: str, var_type: str):
    """Add variable declarations to the builder."""
    if var_type == "input":
        return  # Input variables are function parameters and should not be redeclared

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
