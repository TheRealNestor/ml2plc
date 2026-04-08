"""
PLC type system utilities.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PLCTypeInfo:
    """Information about a PLC type."""

    plc_name: str
    numpy_dtype: Optional[np.dtype]
    onnx_name: str
    size_bytes: int
    is_float: bool = False

    @property
    def limits(self) -> Tuple[int, int]:
        """Get min/max values for this type."""
        if self.is_float or self.numpy_dtype is None:
            return (0, 0)
        info = np.iinfo(self.numpy_dtype)
        return (info.min, info.max)


# Unified type registry
PLC_TYPES: Dict[str, PLCTypeInfo] = {
    "SINT": PLCTypeInfo("SINT", np.int8, "TensorProto.INT8", 1),
    "USINT": PLCTypeInfo("USINT", np.uint8, "TensorProto.UINT8", 1),
    "INT": PLCTypeInfo("INT", np.int16, "TensorProto.INT16", 2),
    "UINT": PLCTypeInfo("UINT", np.uint16, "TensorProto.UINT16", 2),
    "DINT": PLCTypeInfo("DINT", np.int32, "TensorProto.INT32", 4),
    "UDINT": PLCTypeInfo("UDINT", np.uint32, "TensorProto.UINT32", 4),
    "LINT": PLCTypeInfo("LINT", np.int64, "TensorProto.INT64", 8),
    "REAL": PLCTypeInfo("REAL", np.float32, "TensorProto.FLOAT", 4, is_float=True),
    "LREAL": PLCTypeInfo("LREAL", np.float64, "TensorProto.DOUBLE", 8, is_float=True),
}

# Create reverse lookup dictionaries
_ONNX_TO_PLC: Dict[str, str] = {}
for name, info in PLC_TYPES.items():
    full_key = info.onnx_name.lower()  # "tensorproto.int32"
    bare_key = info.onnx_name.split(".")[-1].lower()  # "int32"
    _ONNX_TO_PLC[full_key] = name
    _ONNX_TO_PLC[bare_key] = name

_NUMPY_TO_PLC = {
    info.numpy_dtype: name
    for name, info in PLC_TYPES.items()
    if info.numpy_dtype is not None
}


def plc_type_from_onnx_dtype(dtype: str) -> str:
    """Map ONNX data type strings to PLC data types."""
    if dtype is None:
        raise ValueError("IR layer data type is None; tensor_info is incomplete.")

    dtype_lower = dtype.lower()
    if dtype_lower in _ONNX_TO_PLC:
        return _ONNX_TO_PLC[dtype_lower]

    raise NotImplementedError(f"Data type {dtype} is not supported.")


def numpy_to_plc_type(dtype: np.dtype) -> str:
    """Convert numpy dtype to IEC 61131-3 type."""
    if dtype.type not in _NUMPY_TO_PLC:
        raise NotImplementedError(f"Numpy dtype {dtype} is not supported.")

    return _NUMPY_TO_PLC[dtype.type]


def get_type_size_bytes(dtype_str: str) -> int:
    """Get size in bytes for a dtype string (ONNX or PLC format)."""
    # Try direct PLC name lookup
    if dtype_str in PLC_TYPES:
        return PLC_TYPES[dtype_str].size_bytes

    # Try ONNX format
    dtype_lower = dtype_str.lower()
    if dtype_lower in _ONNX_TO_PLC:
        plc_name = _ONNX_TO_PLC[dtype_lower]
        return PLC_TYPES[plc_name].size_bytes

    # If neither we can use a default size or throw an exception (probably safer option)
    return 4


def numpy_to_plc_cast_func(np_dtype: np.dtype, target_plc_type: str) -> str:
    """Get PLC cast function from numpy dtype to target type."""
    source_type = numpy_to_plc_type(np_dtype)
    return get_conversion_func(source_type, target_plc_type)


def get_type_limits(dtype: np.dtype) -> Tuple[int, int]:
    """Get min/max values for a dtype."""
    plc_type = numpy_to_plc_type(dtype)
    return PLC_TYPES[plc_type].limits


def get_type_limits_from_str(plc_type: str) -> Tuple[int, int]:
    """Get min/max values for a PLC type string."""
    if plc_type not in PLC_TYPES:
        raise ValueError(f"Unknown PLC type: {plc_type}")
    return PLC_TYPES[plc_type].limits


def get_conversion_func(from_type: str, to_type: str) -> str:
    """Get PLC type conversion function name."""
    if from_type == to_type:
        return ""  # No conversion needed
    return f"{from_type}_TO_{to_type}"
