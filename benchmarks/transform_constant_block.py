"""
Transform IEC 61131-3 Structured Text files for Straton PLC compatibility.

Straton requires CONSTANT to be declared per-variable:
    CONSTANT VarName : ARRAY[...] OF REAL := [...];

Instead of the standard VAR CONSTANT block:
    VAR CONSTANT
        VarName : ARRAY[...] OF REAL := [...];
    END_VAR

Usage:
    python transform_st_for_straton.py <input.st>
    python transform_st_for_straton.py <directory>   # processes all .st files
"""

import re
import sys
from pathlib import Path


def transform_constant_blocks(source: str) -> str:
    """
    Find all VAR CONSTANT ... END_VAR blocks and convert each variable
    declaration inside them to use the per-variable CONSTANT prefix,
    placed inside a plain VAR ... END_VAR block.
    """
    # Pattern to match VAR CONSTANT ... END_VAR blocks (possibly with
    # leading whitespace).  We use DOTALL so '.' matches newlines.
    block_pattern = re.compile(
        r"^([ \t]*)VAR\s+CONSTANT\s*\n(.*?)\n\1END_VAR",
        re.MULTILINE | re.DOTALL,
    )

    def _rewrite_block(match: re.Match) -> str:
        indent = match.group(1)
        body = match.group(2)

        # Collect variable declarations.  A declaration may span multiple
        # lines (long arrays with line-wrapped initialisers).  We
        # re-assemble logical lines first: any line that does NOT look
        # like the start of a new declaration (identifier : ...) is a
        # continuation of the previous one.
        logical_lines: list[str] = []
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue  # skip blank lines

            # Does this look like the start of a new declaration?
            # Pattern: <identifier> : <type> ...
            if re.match(r"^[A-Za-z_]\w*\s*:", stripped):
                logical_lines.append(stripped)
            else:
                # Continuation of previous declaration
                if logical_lines:
                    logical_lines[-1] += " " + stripped
                # else: stray text before first decl – keep as-is
                else:
                    logical_lines.append(stripped)

        # Build the replacement VAR block
        inner_indent = indent + "    "
        new_lines = [f"{indent}VAR"]
        for decl in logical_lines:
            # Ensure the declaration ends with a semicolon
            decl = decl.rstrip().rstrip(";") + ";"
            new_lines.append(f"{inner_indent}CONSTANT {decl}")
        new_lines.append(f"{indent}END_VAR")
        return "\n".join(new_lines)

    return block_pattern.sub(_rewrite_block, source)


def process_file(input_path: Path) -> Path:
    """Process a single .st file and write the transformed version."""
    source = input_path.read_text(encoding="utf-8")
    transformed = transform_constant_blocks(source)

    output_path = input_path.with_stem(input_path.stem + "_straton")
    output_path.write_text(transformed, encoding="utf-8")
    return output_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.st | directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        out = process_file(target)
        print(f"Transformed: {target} -> {out}")
    elif target.is_dir():
        st_files = sorted(target.glob("*.st"))
        if not st_files:
            print(f"No .st files found in {target}")
            sys.exit(1)
        for f in st_files:
            # Skip already-transformed files
            if f.stem.endswith("_straton"):
                continue
            out = process_file(f)
            print(f"Transformed: {f.name} -> {out.name}")
        print(f"\nProcessed {len(st_files)} file(s).")
    else:
        print(f"Error: {target} is not a file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
