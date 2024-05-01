"""
Module defining Materials Project input set generators.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801

In case of questions, consult @Andrew-S-Rosen, @esoteric-ephemera or @janosh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPScanRelaxSet,
    MPScanStaticSet,
    MPStaticSet,
)

from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


@dataclass
class MPGGARelaxSetGenerator(RelaxSetGenerator):
    """Class to generate MP-compatible VASP GGA relaxation input sets."""

    config_dict: dict = field(default_factory=lambda: MPRelaxSet()._config_dict)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    inherit_incar: bool | None = False


@dataclass
class MPGGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MPStaticSet()._config_dict)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    inherit_incar: bool | None = False


@dataclass
class MPMetaGGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MPScanStaticSet()._config_dict)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4
    inherit_incar: bool | None = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"LWAVE": False}


@dataclass
class MPMetaGGARelaxSetGenerator(RelaxSetGenerator):
    """Class to generate MP-compatible VASP metaGGA relaxation input sets.

    Parameters
    ----------
    config_dict: dict
        The config dict.
    bandgap_tol: float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.
    """

    config_dict: dict = field(default_factory=lambda: MPScanRelaxSet()._config_dict)
    bandgap_tol: float = 1e-4
    auto_ismear: bool = False
    auto_kspacing: bool = True
    inherit_incar: bool | None = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"LWAVE": True}
