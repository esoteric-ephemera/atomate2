from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, Response, job

from atomate2.common.jobs.eos import PostProcessEosEnergy

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
    max_strain: float = 0.02
    number_of_frames: int | None = None
    postprocessor: EOSPostProcessor = field(default_factory=PostProcessEosEnergy)
    _eos_priority: list[str] = field(
        default_factory=lambda: [
            "vinet",
            "birch_murnaghan",
            "birch",
            "pourier_tarantola" "murnaghan",
        ]
    )

    @job
    def make(
        self,
        structure: Structure,
        eos_data: dict | None = None,
        prev_dir: str | Path = None,
    ) -> Flow:
        """Relax alloy structure."""
        if eos_data is None:
            eos_data = {"relax": {k: [] for k in ("energy", "volume", "stress")}}
            self.number_of_frames = (
                self.number_of_frames or self.postprocessor.min_data_points
            )
            new_volumes = [
                (
                    1.0
                    - self.max_strain
                    + 2 * i * self.max_strain / (self.number_of_frames - 1)
                )
                ** 3
                * structure.volume
                for i in range(self.number_of_frames)
            ]

        else:
            self.postprocessor.fit(eos_data)
            eos_data = {
                "relax": {
                    k: self.postprocessor.results["relax"][k]
                    for k in ("energy", "volume", "stress")
                }
            }

            eos_results = None
            for eos_name in self._eos_priority:
                if not self.postprocessor.results["relax"]["EOS"][eos_name].get(
                    "exception"
                ):
                    eos_results = self.postprocessor.results["relax"]["EOS"][
                        eos_name
                    ].copy()
                    break

            if eos_results is None:
                import json

                raise ValueError(
                    f"All EOS fits failed!\nEOS data:\n{json.dumps(eos_data['relax'])}"
                )

            v_min = min(eos_data["relax"]["volume"])
            v_max = max(eos_data["relax"]["volume"])

            if v_min <= eos_results["v0"] and eos_results["v0"] <= v_max:
                equil_vol_struct = structure.copy()
                equil_vol_struct.scale_lattice(eos_results["v0"])
                return equil_vol_struct

            elif eos_results["v0"] > v_max:
                new_volumes = [
                    eos_results["v0"],
                    (1.0 + self.max_strain) ** 3 * eos_results["v0"],
                ]

            elif eos_results["v0"] < v_min:
                new_volumes = [
                    (1.0 - self.max_strain) ** 3 * eos_results["v0"],
                    eos_results["v0"],
                ]

        eos_jobs = []
        for volume in new_volumes:
            new_structure = structure.copy()
            new_structure.scale_lattice(volume)
            relax_job = self.eos_relax_maker.make(
                structure=new_structure, prev_dir=prev_dir
            )
            relax_job.name += f" deformation {len(eos_data['relax']['volume']) + 1}"

            eos_data["relax"]["energy"].append(relax_job.output.output.energy)
            eos_data["relax"]["volume"].append(relax_job.output.structure.volume)
            eos_data["relax"]["stress"].append(relax_job.output.output.stress)

            eos_jobs.append(relax_job)

        recursive_flow = self.make(
            structure=structure,
            eos_data=eos_data,
            prev_dir=prev_dir,
        )

        new_flow = Flow([*eos_jobs, recursive_flow], output=recursive_flow.output)
        return Response(replace=new_flow, output=recursive_flow.output)


@dataclass
class AlloyEquilibrator(Maker):
    """
    Maker to equilibrate an alloy structure.
    """

    name: str = "Alloy relax maker"
    initial_volume_maker: Maker = None
    relaxation_maker: Maker = None
    static_maker: Maker = None

    def make(
        self,
        structure: Structure,
        initial_volume: float | dict[float] = None,
        prev_dir: str | Path = None,
    ) -> Flow:
        jobs: list[Job] = []

        if isinstance(initial_volume, dict):
            # If `initial_volume` specified as a dict, weight
            # initial volumes according to relative stoichiometry
            weights = {
                element: structure.composition[element]
                / sum(structure.composition.values())
                for element in initial_volume
            }
            initial_volume = sum(
                initial_volume[element] * weight for element, weight in weights.items()
            )

        if initial_volume is not None:
            structure.scale_lattice(initial_volume)

        initial_volume_job = self.initial_volume_maker.make(
            structure=structure,
            prev_dir=prev_dir,
        )
        jobs += [initial_volume_job]

        relax_job = self.relaxation_maker.make(
            structure=initial_volume_job.output, prev_dir=prev_dir
        )
        output = relax_job.output
        jobs += [relax_job]

        if self.static_maker:
            static_job = self.static_maker.make(
                structure=relax_job.output.structure, prev_dir=relax_job.output.dir_name
            )
            jobs += [static_job]
            output = static_job.output

        return Flow(jobs, output=output)
