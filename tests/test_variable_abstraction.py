import pytest

from src.codegen.ir_to_st.variable import Variable, ensure_var


def test_scalar_variable():
    v = Variable(name="x", shape=(1,))
    assert v.size == 1
    assert v.is_scalar is True
    assert v.declare_st() == "x : REAL;"

    # VarRef behavior: indexing a scalar should render as the bare name
    ref = v.at(0)
    assert str(ref) == "x"
    assert str(v.scalar()) == "x"


def test_multi_axis_size_one():
    # Current semantics treat flattened size==1 as scalar; ensure behavior
    v = Variable(name="m", shape=(1, 1))
    assert v.size == 1
    # Ensure declaration and rendering remain consistent
    assert v.is_scalar is True
    assert v.declare_st() == "m : REAL;"
    assert str(v.at(0)) == "m"


def test_array_variable():
    v = Variable(name="y", shape=(4,))
    assert v.size == 4
    assert v.is_scalar is False
    assert v.declare_st() == "y : ARRAY[0..3] OF REAL;"

    # Index renders with brackets
    ref = v.at("i")
    assert str(ref) == "y[i]"
    # scalar() returns element access for arrays
    assert str(v.scalar()) == "y[0]"


def test_unresolved_size_raises():
    # shape with non-positive dim should be treated as unresolved
    v = Variable(name="z", shape=(0,))
    assert v.size == 0
    assert v.is_scalar is False
    with pytest.raises(ValueError):
        _ = v.declare_st()


def test_ensure_var():
    v = ensure_var("a", shape_hint=(2,))
    assert isinstance(v, Variable)
    assert v.name == "a"
    assert v.size == 2
