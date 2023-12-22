"""
Module defining Materials Project 2024 workflows.

These are not the atomate 1-compatible flows!
Contact @esoteric-ephemera for questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.mp.mp import MPGGADoubleRelaxStaticMaker
from atomate2.vasp.jobs.mp.mp24 import (
    MP24GGAPreRelaxMaker,
    MP24GGARelaxMaker,
    MP24GGAStaticMaker,
    MP24MetaGGAPreRelaxMaker,
    MP24MetaGGARelaxMaker,
    MP24MetaGGAStaticMaker,
)

if TYPE_CHECKING:
    from jobflow import Maker


@dataclass
class MP24GGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP 2024 GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP 2024 GGA double relax"
    relax_maker1: Maker | None = field(default_factory=MP24GGAPreRelaxMaker)
    relax_maker2: Maker = field(
        default_factory=lambda: MP24GGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )


@dataclass
class MP24MetaGGADoubleRelaxMaker(DoubleRelaxMaker):
    """MP 2024 r2SCAN meta-GGA double relaxation workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to generate the second relaxation.
    """

    name: str = "MP 2024 meta-GGA double relax"
    relax_maker1: Maker | None = field(default_factory=MP24MetaGGAPreRelaxMaker)
    relax_maker2: Maker = field(
        default_factory=lambda: MP24MetaGGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )


@dataclass
class MP24GGADRSMaker(MPGGADoubleRelaxStaticMaker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP 2024 settings.

    Only the middle job performing a PBE relaxation is non-optional.
    DRS = double relax + static.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    """

    name: str = "MP 2024 GGA relax"
    relax_maker: Maker = field(default_factory=MP24GGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MP24GGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )


@dataclass
class MP24MetaGGADRSMaker(MPGGADoubleRelaxStaticMaker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP 2024 settings.

    Only the middle job performing a PBE relaxation is non-optional.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    """

    name: str = "MP 2024 meta-GGA relax"
    relax_maker: Maker = field(default_factory=MP24MetaGGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MP24MetaGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
