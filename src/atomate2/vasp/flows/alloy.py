""" Equilibrate alloy structures from SQS starting points using VASP. """

from __future__ import annotations

from dataclasses import field, dataclass
from jobflow import Maker
from typing import TYPE_CHECKING

from atomate2.common.flows.alloy import InitialVolumeMaker, AlloyEquilibrator
from atomate2.vasp.jobs.alloy import AlloyEosRelaxMaker, AlloyRelaxMaker, AlloyStaticMaker

if TYPE_CHECKING:
    from atomate2.common.jobs.eos import EO

@dataclass
class VaspInitialVolumeMaker(InitialVolumeMaker):
    """
    Estimate volume of alloy from initial guess and EOS fit.
    """
    name: str = "VASP alloy EOS volume determination"
    eos_relax_maker: Maker = field(default_factory=AlloyEosRelaxMaker)

@dataclass
class VaspAlloyEquilibrator(AlloyEquilibrator):
    """ VASP alloy equilibrator. """

    name: str = "VASP alloy relax maker"
    initial_volume_maker : Maker = field(default_factory=VaspInitialVolumeMaker)
    relaxation_maker: Maker = field(default_factory=AlloyRelaxMaker)
    static_maker: Maker = field(default_factory=AlloyStaticMaker)
