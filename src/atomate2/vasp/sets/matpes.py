"""
Module defining MatPES input set generators.

In case of questions, contact @janosh or @shyuep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.vasp.sets import MatPESStaticSet

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


@dataclass
class MatPesGGAStaticSetGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MatPESStaticSet()._config_dict)
    auto_ismear: bool = False
    auto_kspacing: bool = False


@dataclass
class MatPesMetaGGAStaticSetGenerator(MatPesGGAStaticSetGenerator):
    """Class to generate MP-compatible VASP meta-GGA static input sets."""

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
        return {
            "METAGGA": "R2SCAN",
            "ALGO": "ALL",
            "GGA": None,
            "LWAVE": False,
        }  # unset GGA
