from src.codegen.types import ActivationLayer, ActivationType, NetworkIR
from src.codegen.types import BaseLayer
from src.codegen.ir_to_st.codegen_core import translate_ir_to_st
from src.codegen.types import NetworkIR


def make_network(layer):
    layers = {layer.name: layer}
    execution_order = [layer.name]
    tensor_producers = {layer.outputs[0]: layer.name}
    tensor_consumers = {}
    input_tensors = (layer.inputs[0],)
    output_tensors = (layer.outputs[0],)
    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )


def test_translate_generates_correct_bounds_no_batch():
    # Per-sample 1D input (features only)
    layer = ActivationLayer(
        layer_id=0,
        name="layer0",
        op_type="Relu",
        activation=ActivationType.RELU,
        input_size=3,
        output_size=2,
        inputs=("input_data",),
        outputs=("output_data",),
        input_shape=(3,),
        output_shape=(2,),
        input_type="TensorProto.FLOAT",
        output_type="TensorProto.FLOAT",
    )

    ir = make_network(layer)
    st = translate_ir_to_st(ir, fb_name="TestFB")
    assert "input_data : ARRAY[0..2]" in st
    assert "output_data : ARRAY[0..1]" in st


def test_translate_generates_correct_bounds_with_batch():
    # Batch-first (batch, features) should use feature size for array bounds
    layer = ActivationLayer(
        layer_id=0,
        name="layer1",
        op_type="Relu",
        activation=ActivationType.RELU,
        input_size=4,  # feature size (e.g., batch dim excluded)
        output_size=1,
        inputs=("input_data",),
        outputs=("output_data",),
        input_shape=(1, 4),
        output_shape=(1, 1),
        input_type="TensorProto.FLOAT",
        output_type="TensorProto.FLOAT",
    )

    ir = make_network(layer)
    st = translate_ir_to_st(ir, fb_name="TestFB2")
    assert "input_data : ARRAY[0..3]" in st
    # single-element outputs should be emitted as scalars, not ARRAY[0..0]
    assert "output_data : " in st and "ARRAY[0..0]" not in st
