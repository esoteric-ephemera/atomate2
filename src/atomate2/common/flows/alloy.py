from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from dataclasses import dataclass, field

from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import rescale_volume, PostProcessEosPressure

if TYPE_CHECKING:
    from pathlib import Path
    from jobflow import Job

    from pymatgen.core import Structure

    from atomate2.common.jobs.eos import EOSPostProcessor

@dataclass
class InitialVolumeMaker(Maker):
    """
    Estimate volume of alloy from initial guess and EOS fit.
    """
    name: str = "Alloy EOS volume determination"
    eos_relax_maker: Maker = None
    static_maker: Maker = None
    linear_strain: tuple[float, float] = (-0.02, 0.02)
    number_of_frames: int | None = None
    postprocessor: EOSPostProcessor = field(default_factory=PostProcessEosPressure)

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path = None,
    ) -> Flow:
        """Relax alloy structure."""
        
        eos_job = CommonEosMaker(
            name = self.name + " - EOS fit",
            initial_relax_maker = None,
            eos_relax_maker = self.eos_relax_maker,
            static_maker = self.static_maker,
            linear_strain = self.linear_strain,
            number_of_frames = self.number_of_frames or self.postprocessor.min_data_points,
            postprocessor = self.postprocessor
        ).make(structure = structure, prev_dir = prev_dir)

        rescale_job = rescale_volume(structure, eos_job.output["relax"]["EOS"]["v0"])

        return Flow([eos_job,rescale_job], output = rescale_job.output)

class AlloyEquilibrator(Maker):
    """
    Maker to equilibrate an alloy structure.
    """

    name: str = "Alloy relax maker"
    initial_volume_maker : Maker = None
    relaxation_maker: Maker = None
    static_maker: Maker = None

    def make(
        self,
        structure : Structure,
        initial_volume: float | dict[float] = None,
        prev_dir: str | Path = None,
    ) -> Flow:
        
        jobs: list[Job] = []

        if isinstance(initial_volume, dict):
            # If `initial_volume` specified as a dict, weight
            # initial volumes according to relative stoichiometry
            weights = {
                element: structure.composition[element]/sum(
                    structure.composition.values()
                ) for element in initial_volume
            }
            initial_volume = sum(
                initial_volume[element]*weight for 
                element, weight in weights.items()
            )

        if initial_volume:
            initial_volume_rescale = rescale_volume(structure, initial_volume)
            structure = initial_volume_rescale.output
            jobs += [initial_volume_rescale]
        
        initial_volume_job = self.initial_volume_maker.make(
            structure = structure,
            prev_dir = prev_dir,
        )
        jobs += [initial_volume_job]
        
        relax_job = self.relaxation_maker.make(
            structure = initial_volume_job.output,
            prev_dir = prev_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            static_job = self.static_maker.make(
                structure = relax_job.output.structure, 
                prev_dir = relax_job.output.dir_name
            )
            jobs += [static_job]
            output = static_job.output

        return Flow(jobs, output = output)