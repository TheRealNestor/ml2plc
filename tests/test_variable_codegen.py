import re

from codegen.ir_to_st.variable import Variable
from codegen.ir_to_st.utils.copy_helpers import (
    generate_simple_copy,
    generate_scalar_broadcast,
)


def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_simple_copy_scalar_to_scalar_emits_no_indexing():
    a = Variable("a", (1,))
    b = Variable("b", (1,))
    code = generate_simple_copy(a, b)
    s = _clean_whitespace(str(code))

    # Should be a direct assignment, no indexing
    assert "b := a;" in s
    assert "a[" not in s
    assert "b[" not in s


def test_simple_copy_array_to_array_emits_indexing_and_loop():
    a = Variable("a", (4,))
    b = Variable("b", (4,))
    code = generate_simple_copy(a, b)
    s = _clean_whitespace(str(code))

    # Should emit a FOR loop for indices 0..3 and use indexed accesses
    assert "FOR i := 0 TO 3 DO" in s
    assert "b[i] := a[i];" in s or "b[i] := a[i] ;" in s


def test_scalar_broadcast_uses_scalar_source_and_indexed_destination():
    src = Variable("src", (1,))
    dst = Variable("dst", (5,))
    code = generate_scalar_broadcast(src, dst)
    s = _clean_whitespace(str(code))

    # destination must be indexed, source (scalar) must not be indexed
    assert "FOR i := 0 TO 4 DO" in s
    assert "dst[i]" in s
    assert "src[" not in s
