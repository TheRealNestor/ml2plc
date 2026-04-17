from src.translation_validation.st_to_python import translate_st_to_python


def test_translate_preserves_array_input_no_collapse():
    st = """
FUNCTION_BLOCK FB
VAR_INPUT
    input_data : ARRAY[0..1] OF REAL;
END_VAR
END_FUNCTION_BLOCK
"""

    py_code, fn = translate_st_to_python(st)
    # When input is declared as ARRAY, translator must NOT collapse it to scalar
    assert "input_data = input_data[0]" not in py_code


def test_translate_collapses_scalar_input_declaration():
    st = """
FUNCTION_BLOCK FB
VAR_INPUT
    input_data : REAL;
END_VAR
END_FUNCTION_BLOCK
"""

    py_code, fn = translate_st_to_python(st)
    # When input is declared as scalar, the translator should normalize a
    # runtime single-element list into a scalar
    assert "input_data = input_data[0]" in py_code
