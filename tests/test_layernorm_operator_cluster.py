import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir.converter import onnx_to_ir
from codegen.types import BinaryElementwiseLayer
from codegen.ir_to_st.codegen_core import translate_ir_to_st


def _build_layernorm_cluster_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4])

    eps_arr = np.array([1e-5, 1e-5], dtype=np.float32).reshape(1, 2, 1)
    gamma_arr = np.array([1.0, 0.9, 1.1, 1.05], dtype=np.float32)

    eps = helper.make_tensor(
        "eps", TensorProto.FLOAT, [1, 2, 1], eps_arr.flatten().tolist()
    )
    gamma = helper.make_tensor(
        "gamma", TensorProto.FLOAT, [4], gamma_arr.flatten().tolist()
    )

    nodes = [
        helper.make_node(
            "ReduceMean",
            inputs=["x"],
            outputs=["mean"],
            name="mean",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Sub", inputs=["x", "mean"], outputs=["centered"], name="centered"
        ),
        helper.make_node(
            "Mul", inputs=["centered", "centered"], outputs=["sq"], name="sq"
        ),
        helper.make_node(
            "ReduceMean",
            inputs=["sq"],
            outputs=["var"],
            name="var",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Add", inputs=["var", "eps"], outputs=["var_eps"], name="eps_add"
        ),
        helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std"], name="std"),
        helper.make_node(
            "Reciprocal", inputs=["std"], outputs=["inv_std"], name="inv_std"
        ),
        helper.make_node(
            "Mul", inputs=["centered", "inv_std"], outputs=["norm"], name="norm"
        ),
        helper.make_node("Mul", inputs=["norm", "gamma"], outputs=["y"], name="scale"),
    ]

    graph = helper.make_graph(
        nodes, "layernorm_cluster", [x], [y], initializer=[eps, gamma]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="layernorm-cluster-test",
    )
    onnx.checker.check_model(model)
    return model


def test_layernorm_cluster_extracts_supported_ir_layers_and_broadcast_metadata():
    model = _build_layernorm_cluster_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "layernorm_cluster.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        network_ir = onnx_to_ir(analyzer)
        layers = [network_ir.layers[name] for name in network_ir.execution_order]
        op_types = [layer.op_type for layer in layers]

        assert "ReduceMean" in op_types
        assert "Sub" in op_types
        assert "Mul" in op_types
        assert "Sqrt" in op_types
        assert "Reciprocal" in op_types

        norm_mul = next(l for l in layers if l.name == "norm")
        assert isinstance(norm_mul, BinaryElementwiseLayer)
        assert norm_mul.rhs_const is None
        assert norm_mul.rhs_runtime_size == 2

        scale_mul = next(l for l in layers if l.name == "scale")
        assert isinstance(scale_mul, BinaryElementwiseLayer)
        assert scale_mul.rhs_const is not None
        assert scale_mul.rhs_const.size == 4


def test_layernorm_cluster_translates_to_st_with_reduce_and_elementwise_ops():
    model = _build_layernorm_cluster_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "layernorm_cluster.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        network_ir = onnx_to_ir(analyzer)
        st_code = translate_ir_to_st(network_ir, fb_name="LayerNormCluster")

        assert "ReduceMean" in st_code
        assert "SQRT(" in st_code
        assert "1.0 /" in st_code
        assert "rhs_const_" in st_code
        assert "MOD 4" in st_code
        assert "MOD 2" in st_code
