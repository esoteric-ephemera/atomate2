"""
Module defining Materials Project 2024 workflows.

These are not the atomate 1-compatible flows!
Contact @esoteric-ephemera for questions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.mp24 import (
    MP24GGAPreRelaxMaker,
    MP24GGARelaxMaker,
    MP24GGAStaticMaker,
    MP24MetaGGAPreRelaxMaker,
    MP24MetaGGARelaxMaker,
    MP24MetaGGAStaticMaker,
    _clean_up_files,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pymatgen.core import Structure


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
class MP24GGADRSMaker(Maker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP24 settings.

    The initial relaxation is required, the final static is not.
    Final cleanup of WAVECAR files is also optional.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    clean_files : Sequence[str] | None = ("WAVECAR",)
        If a sequence of str, a list of files from all calcs to remove.
        If None, removes no files. Used by default to remove intermediate WAVECARs
        that stabilize convergence.
    """

    name: str = "MP 2024 GGA relax"
    relax_maker: Maker = field(default_factory=MP24GGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MP24GGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
    clean_files: Sequence[str] | None = ("WAVECAR",)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        1, 2 or 3-step flow with required relaxation and optional final static jobs.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing the MP relaxation workflow.
        """
        relax_flow = self.relax_maker.make(structure=structure, prev_dir=prev_dir)
        output = relax_flow.output
        jobs = [relax_flow]

        if self.static_maker:
            # Run a static calculation
            static_job = self.static_maker.make(
                structure=output.structure, prev_dir=output.dir_name
            )
            output = static_job.output
            jobs += [static_job]

        if (self.clean_files is not None) and len(self.clean_files) > 0:
            to_rm = []
            for file_name in self.clean_files:
                to_rm.extend([os.path.join(job.dir_name, file_name) for job in jobs])
            rm_job = _clean_up_files(to_rm, allow_zpath=True)
            jobs += [rm_job]

        return Flow(jobs=jobs, output=output, name=self.name)


@dataclass
class MP24MetaGGADRSMaker(MP24GGADRSMaker):
    """
    Maker to perform a VASP GGA relaxation workflow with MP 2024 settings.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation.
    static_maker : .BaseVaspMaker
        Maker to generate the static calculation before the relaxation.
    clean_files : Sequence[str] | None = ("WAVECAR",)
        If a sequence of str, a list of files from all calcs to remove.
        If None, removes no files. Used by default to remove intermediate WAVECARs
        that stabilize convergence.
    """

    name: str = "MP 2024 meta-GGA relax"
    relax_maker: Maker = field(default_factory=MP24MetaGGADoubleRelaxMaker)
    static_maker: Maker | None = field(
        default_factory=lambda: MP24MetaGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
