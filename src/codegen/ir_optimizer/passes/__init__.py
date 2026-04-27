"""
IR optimization passes.
"""

from .remove_identity import RemoveIdentityPass
from .remove_noop_reshape import RemoveNoOpReshapePass
from .remove_redundant_quant_pairs import RemoveRedundantQuantPairPass
from .remove_weight_dequant import RemoveWeightDequantPass
from .fuse_linear_activation import FuseLinearActivationPass
from .buffer_allocation import BufferAllocationPass
from .remove_dropout import RemoveDropoutPass
from .remove_softmax import RemoveSoftmaxPass
from .insert_quantize import InsertQuantizePass
from .transpose_weights import TransposeWeightsPass
from .dead_variable_elimination import DeadVariableEliminationPass
from .precision_reduction import PrecisionReductionPass
from .index_precomputation import IndexPrecomputationPass
from .constant_folding import ConstantFoldingPass
from .prune_weights import PruneWeightsPass
from .fold_quantized_weights import FoldQuantizedWeightsPass
from .loop_unrolling import LoopUnrollingPass
from .buffer_minimization import BufferMinimizationPass

__all__ = [
    "RemoveIdentityPass",
    "RemoveNoOpReshapePass",
    "RemoveRedundantQuantPairPass",
    "RemoveWeightDequantPass",
    "FuseLinearActivationPass",
    "BufferAllocationPass",
    "RemoveDropoutPass",
    "InsertQuantizePass",
    "TransposeWeightsPass",
    "DeadVariableEliminationPass",
    "PrecisionReductionPass",
    "IndexPrecomputationPass",
    "ConstantFoldingPass",
    "PruneWeightsPass",
    "FoldQuantizedWeightsPass",
    "LoopUnrollingPass",
    "BufferMinimizationPass",
    "RemoveSoftmaxPass",
]
