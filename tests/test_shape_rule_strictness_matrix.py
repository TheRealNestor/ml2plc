import pytest

from codegen.onnx_to_ir.shape import (
    infer_expand_output_shape,
    infer_slice_output_shape,
    infer_transpose_output_shape,
    infer_unsqueeze_output_shape,
)


@pytest.mark.parametrize(
    "shape,perm,strict,raises,expected",
    [
        ((3,), (1,), True, ValueError, None),
        ((3,), (1,), False, None, (3,)),
        ((2, 3), (1, 0), True, None, (3, 2)),
    ],
)
def test_transpose_strictness_matrix(shape, perm, strict, raises, expected):
    if raises:
        with pytest.raises(raises):
            infer_transpose_output_shape(shape, perm, strict=strict, context="T")
    else:
        assert (
            infer_transpose_output_shape(shape, perm, strict=strict, context="T")
            == expected
        )


@pytest.mark.parametrize(
    "shape,axes,strict,raises,expected",
    [
        ((3,), (), True, ValueError, None),
        ((3,), (), False, None, (3,)),
        ((3,), (0,), True, None, (1, 3)),
    ],
)
def test_unsqueeze_strictness_matrix(shape, axes, strict, raises, expected):
    if raises:
        with pytest.raises(raises):
            infer_unsqueeze_output_shape(shape, axes, strict=strict, context="U")
    else:
        assert (
            infer_unsqueeze_output_shape(shape, axes, strict=strict, context="U")
            == expected
        )


@pytest.mark.parametrize(
    "shape,starts,ends,axes,steps,strict,raises",
    [
        ((5,), (), (), (), (), True, ValueError),
        ((5,), (), (), (), (), False, None),
        ((5,), (0,), (5,), (0,), (1,), True, None),
        ((5,), (0,), (5,), (0,), (0,), True, ValueError),
    ],
)
def test_slice_strictness_matrix(shape, starts, ends, axes, steps, strict, raises):
    if raises:
        with pytest.raises(raises):
            infer_slice_output_shape(
                shape,
                starts,
                ends,
                axes,
                steps,
                strict=strict,
                context="S",
            )
    else:
        out = infer_slice_output_shape(
            shape,
            starts,
            ends,
            axes,
            steps,
            strict=strict,
            context="S",
        )
        assert out


@pytest.mark.parametrize(
    "shape,target,strict,raises,expected",
    [
        ((4,), None, True, ValueError, None),
        ((4,), None, False, None, (4,)),
        ((1, 4), (2, 4), True, None, (2, 4)),
        ((4,), (2, 3), True, ValueError, None),
    ],
)
def test_expand_strictness_matrix(shape, target, strict, raises, expected):
    if raises:
        with pytest.raises(raises):
            infer_expand_output_shape(shape, target, strict=strict, context="E")
    else:
        assert (
            infer_expand_output_shape(shape, target, strict=strict, context="E")
            == expected
        )
