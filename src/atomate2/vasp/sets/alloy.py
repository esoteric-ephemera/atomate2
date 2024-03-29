""" Create VASP input sets for alloys. """

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from monty.serialization import loadfn
from typing import TYPE_CHECKING

from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun

_base_param_set = loadfn(
    get_mod_path("atomate2.vasp.sets") / "BaseMPR2SCANRelaxSet.yaml"
)
_base_param_set["POTCAR_FUNCTIONAL"] = "PBE_64"
_base_param_set["POTCAR"].update({"Ba": "Ba_sv_GW", "Dy": "Dy_h", "Er": "Er_h", "Ho": "Ho_h", "Nd": "Nd_h", "Pm": "Pm_h", "Pr": "Pr_h", "Sm": "Sm_h", "Tb": "Tb_h", "Tm": "Tm_h", "Xe": "Xe_GW", "Yb": "Yb_h"})
_base_param_set["INCAR"].update({"METAGGA": None, "GGA": "PS", "LREAL": False, "LMAXMIX": 6, "ISMEAR": 2, "SIGMA": 0.2, "ALGO": "NORMAL", "LELF": False})

@dataclass
class AlloyRelaxSetGenerator(RelaxSetGenerator):
    """Generate PBEsol VASP relaxation input sets for alloys."""

    config_dict: dict = field(default_factory=lambda: _base_param_set)
    auto_ismear: bool = False
    auto_kspacing: bool = False
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
        return {
            "LWAVE": True,
            "LCHARG": False,
            "LAECHG": False,
            "LVTOT": False
        }

@dataclass
class AlloyEosRelaxSetGenerator(RelaxSetGenerator):
    """Generate PBEsol VASP relaxation input sets for alloys."""

    config_dict: dict = field(default_factory=lambda: _base_param_set)
    auto_ismear: bool = False
    auto_kspacing: bool = False
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
        return {
            "ISIF": 2,
            "LWAVE": False,
            "LCHARG": False,
            "LAECHG": False,
            "LVTOT": False
        }


@dataclass
class AlloyStaticSetGenerator(StaticSetGenerator):
    """Generate PBEsol VASP static input sets for alloys."""

    config_dict: dict = field(default_factory=lambda: _base_param_set)
    auto_ismear: bool = False
    auto_kspacing: bool = False
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
        return {
            "ALGO": "FAST",
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "ISMEAR": -5,
        }
