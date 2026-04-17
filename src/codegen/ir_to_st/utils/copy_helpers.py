from ..variable import Variable
from ..st_code import STCode, STCodeBuilder


def generate_simple_copy(
    input_var: Variable,
    output_var: Variable,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    if input_var.is_scalar and output_var.is_scalar:
        builder.add_line(f"{output_var.name} := {input_var.name};")
    else:
        size = max(input_var.size, output_var.size)
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var.at('i')} := {input_var.at('i')};")
        builder.add_line("END_FOR;")
    return builder.build()


def generate_offset_copy(
    input_var: Variable,
    output_var: Variable,
    offset: int,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    size = output_var.size
    builder.add_line(f"FOR i := 0 TO {size - 1} DO")
    with builder.indent():
        if offset == 0:
            builder.add_line(f"{output_var.at('i')} := {input_var.at('i')};")
        elif offset > 0:
            builder.add_line(
                f"{output_var.at('i')} := {input_var.at(f'i + {offset}')}" f";"
            )
        else:
            builder.add_line(
                f"{output_var.at('i')} := {input_var.at(f'i - {-offset}')}" f";"
            )
    builder.add_line("END_FOR;")
    return builder.build()


def generate_strided_copy(
    input_var: Variable,
    output_var: Variable,
    stride: int,
    start_index: int = 0,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    builder.add_line(f"FOR i := 0 TO {output_var.size - 1} DO")
    with builder.indent():
        if start_index == 0:
            builder.add_line(
                f"{output_var.at('i')} := {input_var.at(f'i * {stride}')}" f";"
            )
        else:
            builder.add_line(
                f"{output_var.at('i')} := {input_var.at(f'{start_index} + i * {stride}')}"
                f";"
            )
    builder.add_line("END_FOR;")
    return builder.build()


def generate_scalar_broadcast(
    input_var: Variable,
    output_var: Variable,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    # Use explicit scalar reference
    source = str(input_var.scalar())
    if output_var.is_scalar:
        builder.add_line(f"{output_var.name} := {source};")
    else:
        builder.add_line(f"FOR i := 0 TO {output_var.size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var.at('i')} := {source};")
        builder.add_line("END_FOR;")
    return builder.build()


def generate_modulo_broadcast(
    input_var: Variable,
    output_var: Variable,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    builder.add_line(f"FOR i := 0 TO {output_var.size - 1} DO")
    with builder.indent():
        builder.add_line(
            f"{output_var.at('i')} := {input_var.at(f'i MOD {input_var.size}')}" f";"
        )
    builder.add_line("END_FOR;")
    return builder.build()


def generate_selective_copy(
    input_var: Variable,
    output_var: Variable,
    indices: list,
    comment: str = "",
) -> STCode:
    builder = STCodeBuilder()
    if comment:
        builder.add_line(f"(* {comment} *)")
    if len(indices) <= 16:
        for out_idx, in_idx in enumerate(indices):
            builder.add_line(
                f"{output_var.at(out_idx)} := {input_var.at(int(in_idx))};"
            )
    else:
        builder.add_line(
            f"(* WARNING: large Gather index list, using conservative copy *)"
        )
        builder.add_line(f"FOR i := 0 TO {len(indices) - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var.at('i')} := {input_var.at('i')};")
        builder.add_line("END_FOR;")
    return builder.build()
