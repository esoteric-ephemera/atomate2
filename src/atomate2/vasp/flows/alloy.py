""" Equilibrate alloy structures from SQS starting points using VASP. """

from __future__ import annotations

from dataclasses import field, dataclass
from typing import TYPE_CHECKING

from atomate2.common.flows.alloy import InitialVolumeMaker, AlloyEquilibrator
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.alloy import AlloyEosRelaxMaker, AlloyRelaxMaker, AlloyStaticMaker

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker

@dataclass
class VaspInitialVolumeMaker(InitialVolumeMaker):
    """
    Estimate volume of alloy from initial guess and EOS fit.
    """
    name: str = "VASP alloy EOS volume determination"
    eos_relax_maker: BaseVaspMaker = field(default_factory=AlloyEosRelaxMaker)

@dataclass
class VaspAlloyDoubleRelaxMaker(DoubleRelaxMaker):
    """ Perform a double relaxation using alloy params. """
    name : str = "VASP alloy double relax maker"
    relax_maker1 : BaseVaspMaker = field(default_factory=AlloyRelaxMaker)
    relax_maker2 : BaseVaspMaker = field(default_factory=AlloyRelaxMaker)

@dataclass
class VaspAlloyEquilibrator(AlloyEquilibrator):
    """ VASP alloy equilibrator. """

    name: str = "VASP alloy equilibrator"
    initial_volume_maker : BaseVaspMaker = field(default_factory=VaspInitialVolumeMaker)
    relaxation_maker: BaseVaspMaker = field(default_factory=AlloyRelaxMaker)
    static_maker: BaseVaspMaker = field(default_factory=AlloyStaticMaker)
