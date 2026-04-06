from __future__ import annotations

import sys
from types import ModuleType


class Solver:
    pass


class _SolversModule(ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        solver_type = type(name, (Solver,), {})
        setattr(self, name, solver_type)
        return solver_type


def manual_seed(_seed=None) -> None:
    return None


fluxion = ModuleType("refiners.fluxion")
fluxion_utils = ModuleType("refiners.fluxion.utils")
fluxion_utils.manual_seed = manual_seed
fluxion.utils = fluxion_utils

foundationals = ModuleType("refiners.foundationals")
latent_diffusion = ModuleType("refiners.foundationals.latent_diffusion")
solvers = _SolversModule("refiners.foundationals.latent_diffusion.solvers")
latent_diffusion.Solver = Solver
latent_diffusion.solvers = solvers
foundationals.latent_diffusion = latent_diffusion

sys.modules.setdefault("refiners.fluxion", fluxion)
sys.modules.setdefault("refiners.fluxion.utils", fluxion_utils)
sys.modules.setdefault("refiners.foundationals", foundationals)
sys.modules.setdefault(
    "refiners.foundationals.latent_diffusion", latent_diffusion
)
sys.modules.setdefault(
    "refiners.foundationals.latent_diffusion.solvers", solvers
)

__all__ = ("Solver", "foundationals", "fluxion", "manual_seed")
