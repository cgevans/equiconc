from . import equiconc as _native
from .equiconc import *

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
