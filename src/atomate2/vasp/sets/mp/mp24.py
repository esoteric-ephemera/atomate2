"""
Module defining Materials Project 2024 parameters.

These are not the atomate 1-compatible sets!
Contact @esoteric-ephemera for questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files as IRF
from typing import TYPE_CHECKING

from monty.serialization import loadfn

from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun

_BASE_MP24_RELAX_SET = loadfn(IRF("atomate2.vasp.sets") / "BaseMP24RelaxSet.yaml")


@dataclass
class MP24GGARelaxSetGenerator(RelaxSetGenerator):
    """Class to generate MP 2024-compatible VASP GGA relax input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP24_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4

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
            "GGA": "PE",
            "METAGGA": None,
        }


@dataclass
class MP24GGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP 2024-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP24_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4

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
            "GGA": "PE",
            "METAGGA": None,
            "ALGO": "FAST",
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "ISMEAR": -5,
        }


@dataclass
class MP24MetaGGARelaxSetGenerator(RelaxSetGenerator):
    """Class to generate MP 2024-compatible VASP meta-GGA relax input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP24_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4

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
        return {"LCHARG": True, "LWAVE": True, "GGA": None}


@dataclass
class MP24MetaGGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP 2024-compatible VASP meta-GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP24_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4

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
            "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
        }
