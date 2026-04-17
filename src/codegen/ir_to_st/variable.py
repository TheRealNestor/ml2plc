from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Variable:
    name: str
    shape: tuple[int, ...]
    plc_type: str = "REAL"

    @property
    def size(self) -> int:
        # Return the flattened size of the variable. If any dimension is
        # non-positive or non-integer (ambiguous/unresolved) return 0 so
        # callers can detect unresolved sizes instead of silently
        # treating them as length-1 scalars.
        s = 1
        for d in self.shape:
            if not isinstance(d, int) or d <= 0:
                return 0
            s *= d
        return s

    @property
    def is_scalar(self) -> bool:
        # A variable is considered scalar when its flattened size is 1.
        # This preserves previous generator semantics where shape (1,) was
        # treated as a scalar for presentation (assignments / broadcasts).
        # Note: size==0 signals unresolved dims and should be handled by
        # callers before emission.
        return self.size == 1

    def at(self, index: int | str) -> "VarRef":
        """Return a VarRef representing this variable indexed by `index`.

        Examples:
            var.at('i') -> renders as "name[i]"
            var.at('0') -> renders as "name[0]"
        """
        return VarRef(var=self, index=_coerce_index(index))

    def scalar(self) -> "VarRef":
        """Return a VarRef representing the scalar view of this Variable.

        For true scalars this renders as the bare variable name. For arrays
        it returns element 0 (i.e., name[0]).
        """
        if self.is_scalar:
            return VarRef(var=self, index=None)
        # For arrays (even length-1), scalar() returns element 0.
        return VarRef(var=self, index=0)

    def declare_st(self) -> str:
        # If the size is unresolved (0) emit a clear error rather than
        # producing an invalid ARRAY declaration like [0..-1]. Callers
        # should validate/resolve shapes before code emission.
        sz = self.size
        if self.is_scalar:
            return f"{self.name} : {self.plc_type};"
        if sz == 0:
            raise ValueError(
                f"Cannot declare variable '{self.name}' with unresolved size {self.shape}"
            )
        return f"{self.name} : ARRAY[0..{sz - 1}] OF {self.plc_type};"


def ensure_var(var, shape_hint: tuple | None = None):
    """Ensure the object is a Variable. If given a string, construct a Variable
    with an optional shape hint or default scalar shape.
    """
    if isinstance(var, Variable):
        return var
    shape = tuple(shape_hint) if shape_hint else (1,)
    return Variable(name=var, shape=shape)


@dataclass(frozen=True)
class VarRef:
    """Lightweight reference object for a variable or an indexed element.

    This encapsulates presentation concerns for variable references so
    generators can build expressions using VarRef objects instead of raw
    strings. VarRef renders to ST by returning either 'name' or
    'name[index]'.
    """

    var: Variable
    index: int | str | None = None

    def __str__(self) -> str:
        # When rendering a VarRef, prefer the scalar name when the
        # underlying Variable is considered scalar. This keeps generated
        # code safe when a generator calls .at(...) on a variable whose
        # flattened size is 1 (e.g., shape (1,) or unresolved single
        # element). In that case indexing would produce invalid ST like
        # `x[0]` for a scalar `x` declared as `x : REAL;` — instead render
        # simply `x`.
        if self.index is None:
            return self.var.name
        if self.var.is_scalar:
            # Render as scalar (ignore index) to avoid 'index into scalar'
            # errors in downstream code. Generators should prefer the
            # appropriate VarRef form, but this is a safe fallback.
            return self.var.name
        return f"{self.var.name}[{self.index}]"


def _coerce_index(idx: int | str | None) -> int | str | None:
    # Keep as-is; callers may pass integers or string expressions like 'i'.
    return idx
