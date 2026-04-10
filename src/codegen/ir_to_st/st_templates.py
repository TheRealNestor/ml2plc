"""
Structured Text language constructs and templates.

High-level ST patterns organized by category:
  - Variable declarations and constants
  - Control flow (FOR loops, IF blocks)
  - Variable sections (VAR, VAR_INPUT, etc.)
  - Function blocks
  - Program wrappers
  - Configuration templates
"""

from typing import Optional, Union, Tuple, List
import numpy as np

from .st_code import STCode, STCodeBuilder


# =============================================================================
# Variable Declarations and Constants
# =============================================================================


def st_section_header(name: str) -> STCode:
    """Create a section header comment (e.g., '(* Layer Computations *)')"""
    return STCode.from_lines(f"(* {name} *)")


def st_var_declaration(
    name: str,
    var_type: str,
    dimensions: Optional[Union[int, Tuple[int, ...]]] = None,
    init_value: Optional[str] = None,
) -> STCode:
    """
    Generate a variable declaration.

    Args:
        name: Variable name
        var_type: PLC type (e.g., "REAL", "DINT", "ARRAY")
        dimensions: If int, creates ARRAY[0..n] OF var_type
                   If tuple, creates ARRAY[0..n1, 0..n2, ...] OF var_type
        init_value: Optional initialization value

    Returns:
        STCode with declaration
    """
    if dimensions is not None:
        if isinstance(dimensions, int):
            decl = f"{name} : ARRAY[0..{dimensions-1}] OF {var_type}"
        else:
            array_dims = ", ".join(f"0..{d-1}" for d in dimensions)
            decl = f"{name} : ARRAY[{array_dims}] OF {var_type}"
    else:
        decl = f"{name} : {var_type}"

    if init_value:
        decl += f" := {init_value}"

    return STCode.from_lines(f"{decl};")


def st_array_constant(
    name: str, values: np.ndarray, plc_type: str, is_integer: bool = False
) -> STCode:
    """
    Generate an array constant declaration.

    Args:
        name: Constant name
        values: NumPy array of values
        plc_type: PLC type for array elements
        is_integer: Whether to format as integers

    Returns:
        STCode with constant declaration
    """
    flat_values = values.flatten()
    if is_integer:
        value_strs = [str(int(v)) for v in flat_values]
    else:
        value_strs = [str(float(v)) for v in flat_values]

    values_str = ", ".join(value_strs)
    return STCode.from_lines(
        f"{name} : ARRAY[0..{len(flat_values)-1}] OF {plc_type} := [{values_str}];"
    )


def st_scalar_constant(
    name: str, value: Union[float, int], plc_type: str, is_integer: bool = False
) -> STCode:
    """
    Generate a scalar constant declaration.

    Args:
        name: Constant name
        value: Scalar value
        plc_type: PLC type
        is_integer: Whether to format as integer

    Returns:
        STCode with constant declaration
    """
    if is_integer:
        value_str = str(int(value))
    else:
        value_str = str(float(value))

    return STCode.from_lines(f"{name} : {plc_type} := {value_str};")


# =============================================================================
# Control Flow Constructs
# =============================================================================


def st_for_loop(
    index_var: str,
    start: Union[int, str],
    end: Union[int, str],
    body: Optional[STCode] = None,
) -> STCode:
    """
    Generate a FOR loop structure.

    Args:
        index_var: Loop counter variable name
        start: Start value (int, string literal, or expression) - inclusive
        end: End value (int, string literal, or expression) - inclusive
        body: Optional body code (will be indented)

    Returns:
        STCode with FOR loop
    """
    builder = STCodeBuilder()
    builder.add_line(f"FOR {index_var} := {start} TO {end} DO")

    if body:
        builder.add_code(body.indent())

    builder.add_line("END_FOR;")
    return builder.build()


def st_if_block(
    condition: str,
    then_body: Optional[STCode] = None,
    else_body: Optional[STCode] = None,
) -> STCode:
    """
    Generate an IF block structure.

    Args:
        condition: IF condition expression
        then_body: Optional THEN body (will be indented)
        else_body: Optional ELSE body (will be indented)

    Returns:
        STCode with IF block
    """
    builder = STCodeBuilder()
    builder.add_line(f"IF {condition} THEN")

    if then_body:
        builder.add_code(then_body.indent())

    if else_body:
        builder.add_line("ELSE")
        builder.add_code(else_body.indent())

    builder.add_line("END_IF;")
    return builder.build()


# =============================================================================
# Variable Sections (VAR declarations)
# =============================================================================


def st_var_section_header(section_name: str = "VAR") -> STCode:
    """Generate VAR section header (e.g., 'VAR', 'VAR_INPUT', 'VAR_OUTPUT', 'VAR CONSTANT')."""
    return STCode.from_lines(section_name)


def st_var_section_footer() -> STCode:
    """Generate VAR section footer."""
    return STCode.from_lines("END_VAR", "")


def st_var_section(
    section_name: str, declarations: List[STCode], add_footer: bool = True
) -> STCode:
    """
    Generate a complete VAR section.

    Args:
        section_name: Section type ('VAR', 'VAR_INPUT', 'VAR_OUTPUT', 'VAR CONSTANT')
        declarations: List of STCode objects for each declaration
        add_footer: Whether to include END_VAR

    Returns:
        STCode with complete section
    """
    builder = STCodeBuilder()
    builder.add_line(section_name)

    for decl in declarations:
        builder.add_code(decl.indent())
        builder.add_line()  # Blank line between declarations

    if add_footer:
        builder.add_line("END_VAR")
        builder.add_line()  # Blank line after section

    return builder.build()


# =============================================================================
# Function Block Structures
# =============================================================================


def st_function_block_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def st_function_block_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK", "")


# =============================================================================
# Program Wrappers and Configurations
# =============================================================================


def generate_program_wrapper(
    fb_name: str, program_name: str = "prog0", instance_name: str = "nn"
) -> STCode:
    """Generate a PROGRAM wrapper that instantiates and calls the function block."""
    return STCode.from_lines(
        f"PROGRAM {program_name}",
        "VAR",
        f"    {instance_name} : {fb_name};",
        "END_VAR",
        "",
        f"{instance_name}();",
        "",
        "END_PROGRAM",
        "",
    )


def generate_openplc_configuration(
    program_name: str = "prog0",
    configuration_name: str = "Config0",
    resource_name: str = "Res0",
    task_name: str = "Main",
    task_interval: str = "T#1000ms",
    task_priority: int = 0,
    instance_name: str = "Inst0",
) -> STCode:
    """Generate OpenPLC configuration footer (CONFIGURATION / RESOURCE / TASK mapping)."""
    return STCode.from_lines(
        f"CONFIGURATION {configuration_name}",
        "",
        f"  RESOURCE {resource_name} ON PLC",
        f"    TASK {task_name}(INTERVAL := {task_interval},PRIORITY := {task_priority});",
        f"    PROGRAM {instance_name} WITH {task_name} : {program_name};",
        "  END_RESOURCE",
        "END_CONFIGURATION",
        "",
    )
