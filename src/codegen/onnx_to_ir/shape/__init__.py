"""Dedicated shape inference package decoupled from core ONNX->IR passes."""

from . import api as _api

__all__ = list(_api.__all__)
globals().update({name: getattr(_api, name) for name in __all__})

del _api
