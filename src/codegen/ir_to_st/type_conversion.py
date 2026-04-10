"""
PLC type system utilities.

Supports conversion between:
- ONNX types (numeric constants: 1=FLOAT, 7=INT64, etc.)
- ONNX string names (e.g., "TensorProto.FLOAT")
- Canonical internal representation
- IEC 61131-3 PLC types (REAL, LREAL, INT, etc.)
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import IntEnum


# ONNX TensorProto type constants (from onnx.TensorProto)
# https://github.com/onnx/onnx/blob/main/onnx/defs/schema.h
class OnnxDataType(IntEnum):
    """ONNX TensorProto data type constants"""

    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16


@dataclass
class PLCTypeInfo:
    """Information about a PLC type."""

    plc_name: str
    numpy_dtype: Optional[np.dtype]
    onnx_type: OnnxDataType  # Primary ONNX type constant
    size_bytes: int
    is_float: bool = False

    @property
    def limits(self) -> Tuple[int, int]:
        """Get min/max values for this type."""
        if self.is_float or self.numpy_dtype is None:
            return (0, 0)
        info = np.iinfo(self.numpy_dtype)
        return (info.min, info.max)


# PLC types supported by this compiler (subset of ONNX types)
PLC_TYPES: Dict[str, PLCTypeInfo] = {
    "SINT": PLCTypeInfo("SINT", np.int8, OnnxDataType.INT8, 1),
    "USINT": PLCTypeInfo("USINT", np.uint8, OnnxDataType.UINT8, 1),
    "INT": PLCTypeInfo("INT", np.int16, OnnxDataType.INT16, 2),
    "UINT": PLCTypeInfo("UINT", np.uint16, OnnxDataType.UINT16, 2),
    "DINT": PLCTypeInfo("DINT", np.int32, OnnxDataType.INT32, 4),
    "UDINT": PLCTypeInfo("UDINT", np.uint32, OnnxDataType.UINT32, 4),
    "LINT": PLCTypeInfo("LINT", np.int64, OnnxDataType.INT64, 8),
    "REAL": PLCTypeInfo("REAL", np.float32, OnnxDataType.FLOAT, 4, is_float=True),
    "LREAL": PLCTypeInfo("LREAL", np.float64, OnnxDataType.DOUBLE, 8, is_float=True),
}

# Build reverse lookup: ONNX type -> PLC type
_ONNX_TYPE_TO_PLC: Dict[int, str] = {
    int(info.onnx_type): plc_name for plc_name, info in PLC_TYPES.items()
}

# Build string lookup for common formats
_ONNX_STRING_TO_TYPE: Dict[str, int] = {
    "FLOAT": OnnxDataType.FLOAT,
    "INT8": OnnxDataType.INT8,
    "UINT8": OnnxDataType.UINT8,
    "INT16": OnnxDataType.INT16,
    "UINT16": OnnxDataType.UINT16,
    "INT32": OnnxDataType.INT32,
    "UINT32": OnnxDataType.UINT32,
    "INT64": OnnxDataType.INT64,
    "UINT64": OnnxDataType.UINT64,
    "DOUBLE": OnnxDataType.DOUBLE,
    "BOOL": OnnxDataType.BOOL,
    "FLOAT16": OnnxDataType.FLOAT16,
    "BFLOAT16": OnnxDataType.BFLOAT16,
    "STRING": OnnxDataType.STRING,
    # TensorProto.* format
    "TENSORPROTO.FLOAT": OnnxDataType.FLOAT,
    "TENSORPROTO.INT8": OnnxDataType.INT8,
    "TENSORPROTO.UINT8": OnnxDataType.UINT8,
    "TENSORPROTO.INT16": OnnxDataType.INT16,
    "TENSORPROTO.UINT16": OnnxDataType.UINT16,
    "TENSORPROTO.INT32": OnnxDataType.INT32,
    "TENSORPROTO.UINT32": OnnxDataType.UINT32,
    "TENSORPROTO.INT64": OnnxDataType.INT64,
    "TENSORPROTO.UINT64": OnnxDataType.UINT64,
    "TENSORPROTO.DOUBLE": OnnxDataType.DOUBLE,
    "TENSORPROTO.BOOL": OnnxDataType.BOOL,
    "TENSORPROTO.FLOAT16": OnnxDataType.FLOAT16,
    "TENSORPROTO.BFLOAT16": OnnxDataType.BFLOAT16,
    "TENSORPROTO.STRING": OnnxDataType.STRING,
}

_NUMPY_TO_PLC = {
    info.numpy_dtype: plc_name
    for plc_name, info in PLC_TYPES.items()
    if info.numpy_dtype is not None
}


def normalize_onnx_type(dtype: Union[int, str]) -> int:
    """
    Convert various ONNX type representations to numeric type constant.

    Accepts:
    - int: Direct ONNX type constant (1=FLOAT, 7=INT64, etc.)
    - str: "FLOAT", "INT64", "TensorProto.FLOAT", etc.

    Returns:
        int: ONNX type constant (0-26)

    Raises:
        ValueError: If type is unknown or unsupported
    """
    if isinstance(dtype, int):
        try:
            return int(OnnxDataType(dtype))
        except ValueError:
            raise ValueError(f"Unknown ONNX type constant: {dtype}")

    if isinstance(dtype, str):
        dtype_upper = dtype.upper()
        if dtype_upper in _ONNX_STRING_TO_TYPE:
            return int(_ONNX_STRING_TO_TYPE[dtype_upper])
        raise ValueError(f"Unknown ONNX type string: {dtype}")

    raise TypeError(f"ONNX type must be int or str, got {type(dtype)}")


def plc_type_from_onnx_dtype(dtype: Union[int, str]) -> str:
    """
    Map ONNX data type to PLC data type.

    Args:
        dtype: ONNX type (int constant or string name)

    Returns:
        str: PLC type name (REAL, LREAL, INT, DINT, etc.)

    Raises:
        ValueError: If ONNX type is unknown
        NotImplementedError: If ONNX type is not supported in PLC
    """
    if dtype is None:
        raise ValueError("Data type cannot be None")

    # Normalize to ONNX int constant
    onnx_type = normalize_onnx_type(dtype)

    # Check if we support this type
    if onnx_type not in _ONNX_TYPE_TO_PLC:
        raise NotImplementedError(
            f"ONNX type {OnnxDataType(onnx_type).name} (constant {onnx_type}) "
            f"is not supported. Supported types: {list(_ONNX_TYPE_TO_PLC.values())}"
        )

    return _ONNX_TYPE_TO_PLC[onnx_type]


def numpy_to_plc_type(dtype: np.dtype) -> str:
    """Convert numpy dtype to IEC 61131-3 type."""
    if dtype.type not in _NUMPY_TO_PLC:
        raise NotImplementedError(f"Numpy dtype {dtype} is not supported.")

    return _NUMPY_TO_PLC[dtype.type]


def get_type_size_bytes(dtype_str: Union[int, str]) -> int:
    """
    Get size in bytes for a dtype.

    Args:
        dtype_str: ONNX type (int/str) or PLC type name

    Returns:
        int: Size in bytes

    Raises:
        ValueError: If type is unknown
    """
    # Try PLC name lookup first
    if isinstance(dtype_str, str) and dtype_str in PLC_TYPES:
        return PLC_TYPES[dtype_str].size_bytes

    # Try ONNX format conversion
    try:
        plc_name = plc_type_from_onnx_dtype(dtype_str)
        return PLC_TYPES[plc_name].size_bytes
    except (ValueError, NotImplementedError):
        raise ValueError(
            f"Cannot determine size for type {dtype_str}. "
            f"Must be a valid PLC type or ONNX type."
        )


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
