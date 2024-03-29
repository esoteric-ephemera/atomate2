""" Flows to relax an alloy structure with a forcefield. """
from __future__ import annotations
from atomate2.common.flows.alloy import InitialVolumeMaker, AlloyEquilibrator

from dataclasses import dataclass, field

from atomate2.forcefields import MLFF
from atomate2.forcefields.jobs import CHGNetRelaxMaker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobflow import Maker

@dataclass
class CHGNetInitialVolumeMaker(InitialVolumeMaker):
    name : str = f"{MLFF.CHGNet} EOS initial volume maker"
    eos_relax_maker : Maker = field(
        default_factory = lambda : CHGNetRelaxMaker(
            relax_cell = False,
        )
    )
    static_maker : Maker = None

@dataclass
class CHGNetAlloyEquilibrator(AlloyEquilibrator):
    name : str = f"{MLFF.CHGNet} alloy relax maker"
    initial_volume_maker : Maker = field(default_factory=CHGNetInitialVolumeMaker)
    relaxation_maker : Maker = field(default_factory=CHGNetRelaxMaker)
    static_maker : Maker = None
