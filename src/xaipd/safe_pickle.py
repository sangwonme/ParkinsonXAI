"""Restricted unpickling for untrusted data pickles (CARE-PD).
1) `scan_globals(path)` lists every GLOBAL/STACK_GLOBAL the pickle would import, WITHOUT executing it.
2) `safe_load(path)` unpickles with an allowlist (numpy array reconstruction + plain containers);
   anything else raises UnpicklingError instead of being imported."""
from __future__ import annotations

import io
import pickle
import pickletools

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    # protocol-5 array reconstruction: builds an ndarray from a raw byte buffer (no code execution)
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "set"),
    ("builtins", "frozenset"), ("builtins", "str"), ("builtins", "int"), ("builtins", "float"),
    ("builtins", "bytes"), ("builtins", "complex"), ("builtins", "bool"),
}


def scan_globals(path) -> set[tuple[str, str]]:
    """Static opcode scan. STACK_GLOBAL operands come from the two preceding string pushes."""
    found, strings = set(), []
    with open(path, "rb") as f:
        data = f.read()
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        elif op.name == "MEMOIZE":
            pass
        elif op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            found.add((mod, name))
        elif op.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]) if len(strings) >= 2 else ("?", "?"))
    return found


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def safe_load(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()


class ChStub:
    """Inert stand-in for chumpy.ch.Ch: only stores the pickled state dict; no chumpy code runs."""

    def __setstate__(self, state):
        self.state = state

    @property
    def array(self):
        return self.state["x"] if isinstance(self.state, dict) and "x" in self.state else None


class _SMPLRestricted(_Restricted):
    """For the legacy (py2) SMPL model pickle: chumpy.Ch -> inert stub, scipy csc_matrix allowed,
    __builtin__.set -> builtins.set. Everything else still goes through the allowlist."""

    def find_class(self, module, name):
        if (module, name) == ("chumpy.ch", "Ch"):
            return ChStub
        if (module, name) in (("scipy.sparse.csc", "csc_matrix"), ("scipy.sparse._csc", "csc_matrix")):
            from scipy.sparse import csc_matrix
            return csc_matrix
        if module == "__builtin__":
            module = "builtins"
        return super().find_class(module, name)


def safe_load_smpl(path):
    with open(path, "rb") as f:
        d = _SMPLRestricted(f, encoding="latin1").load()
    return {k: (v.array if isinstance(v, ChStub) else v) for k, v in d.items()}
