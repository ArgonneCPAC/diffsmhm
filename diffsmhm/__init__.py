"""Top-level imports for differentiable_smhm package."""
# flake8: noqa

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    del jax
except Exception:
    raise RuntimeError(
        "Unable to automatically enable JAX for 64-bit types!"
    )
from .galhalo_models import *
from .galhalo_models import default_model_params
