"""
Helpers to represent and manipulate Structured Text code snippets.

This module centralizes all Structured Text code generation utilities, including:
- STCode: Immutable representation of ST code blocks
- STCodeBuilder: Chainable builder with indentation management
- Common ST constructs: declarations, sections, comments, control flow
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List
from contextlib import contextmanager
import numpy as np


@dataclass(frozen=True)
class STCode:
    """Represents a piece of Structured Text code"""

    lines: Tuple[str, ...]

    def __str__(self) -> str:
        """Return code as properly formatted string with newlines."""
        return "\n".join(self.lines)

    def __repr__(self) -> str:
        """Return repr for debugging."""
        return f"STCode({len(self.lines)} lines)"

    def __add__(self, other: "STCode") -> "STCode":
        """Combine two code blocks"""
        return STCode(self.lines + other.lines)

    def indent(self, level: int = 1) -> "STCode":
        """Return indented version of code"""
        indent_str = "    " * level
        return STCode(tuple(indent_str + line if line else line for line in self.lines))

    def to_string(self) -> str:
        """Convert to string"""
        return "\n".join(self.lines)

    @staticmethod
    def from_lines(*lines: str) -> "STCode":
        """Create from individual lines"""
        return STCode(tuple(lines))

    @staticmethod
    def empty() -> "STCode":
        """Create empty code block"""
        return STCode(())

    @staticmethod
    def blank_line() -> "STCode":
        """Create blank line"""
        return STCode(("",))


@dataclass
class STCodeBuilder:
    """Helper for building ST code with automatic indentation tracking."""

    _code: STCode = field(default_factory=STCode.empty)
    _indent_level: int = 0

    def add_line(self, line: str = "") -> "STCodeBuilder":
        """Add a single indented line and return self for chaining."""
        if line:
            self._code += STCode.from_lines("    " * self._indent_level + line)
        else:
            self._code += STCode.blank_line()
        return self

    def add_lines(self, *lines: str) -> "STCodeBuilder":
        """Add multiple indented lines and return self for chaining."""
        for line in lines:
            self.add_line(line)
        return self

    def add_code(self, code: STCode) -> "STCodeBuilder":
        """Add pre-built code block with current indentation and return self for chaining."""
        self._code += code.indent(self._indent_level)
        return self

    def __iadd__(self, code: STCode) -> "STCodeBuilder":
        """Support += operator for adding STCode directly."""
        return self.add_code(code)

    @contextmanager
    def indent(self):
        """Context manager for indented blocks."""
        self._indent_level += 1
        try:
            yield self
        finally:
            self._indent_level -= 1

    def add_comment(self, text: str) -> "STCodeBuilder":
        """Add a single-line comment and return self for chaining."""
        return self.add_line(f"(* {text} *)")

    def build(self) -> STCode:
        """Get the final STCode."""
        return self._code


# ============================================================================
# Common ST Constructs
# ============================================================================


def st_comment(text: str) -> STCode:
    """Create a single-line ST comment."""
    return STCode.from_lines(f"(* {text} *)")


def st_multiline_comment(lines: List[str]) -> STCode:
    """Create a multi-line ST comment block."""
    if not lines:
        return STCode.empty()
    return STCode.from_lines("(*", *[f"  {line}" for line in lines], "*)")


def st_section_header(name: str) -> STCode:
    """Create a section header comment (e.g., '(* Layer Computations *)')"""
    return st_comment(f" {name} ")


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


def st_function_block_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def st_function_block_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK", "")


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
