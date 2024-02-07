from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Flow, Maker, job
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

try:
    from atomate2.common.flows.eos import CommonEosMaker
    from atomate2.common.jobs.eos import apply_strain_to_structure

except ImportError:
    CommonEosMaker = None

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pymatgen.core import Structure


class FindInitialVolume(Maker):
    """
    Estimate volume of alloy from initial guess and EOS fit.
    """

    name: str = "Binary alloy EOS volume determination"
    eos_relax_maker: Maker = None
    static_maker: Maker = None

    def make(
        self,
        structure: Structure,
        initial_volume: float,
        strain: float | Sequence[float] | None = None,
        number_of_frames: int = 3,
        prev_dir: str | Path = None,
    ) -> Flow:
        """Relax alloy structure."""
        jobs: list[job] = []

        # rescale structure to have specified initial guess volume
        scaling = (structure.volume / initial_volume) ** (1.0 / 3.0)

        deformation_matrix = [
            [scaling if i == j else 0.0 for j in range(3)] for i in range(3)
        ]

        dst = DeformStructureTransformation(deformation=deformation_matrix)
        structure = dst.apply_transformation(structure)

        # purposefully asymmetric to avoid applying zero strain in default case
        strain = strain or (-0.019, 0.021)
        if isinstance(strain, float):
            linear_strain = (-abs(strain), abs(strain))
        else:
            linear_strain = strain

        eos_flow = CommonEosMaker(
            name=self.name,
            initial_relax_maker=None,
            eos_relax_maker=self.eos_relax_maker,
            static_maker=self.static_maker,
            strain=linear_strain,
            number_of_frames=number_of_frames,
        ).make(structure=structure, prev_dir=prev_dir)

        return


class _BinaryAlloyRelaxation(Maker):
    """
    Maker to relax a binary alloy SQS structure.
    """

    name: str = "Binary alloy relaxation"
    deformation_maker: Maker = None
    relaxation_maker: Maker = None
    static_maker: Maker = None
    sqs_required_kwargs: tuple[str, ...] = (
        "symmetry",
        "site_composition",
        "nrun",
        "sqs_scaling",
    )

    def make(
        self,
        site_composition: dict[str, float],
        endpoint_volumes: list[float],
        SQS_structure: Structure = None,
        alloy_sqs_kwargs: dict = None,
        prev_dir: str | Path = None,
    ) -> Flow:
        jobs: list[job] = []

        if alloy_sqs_kwargs.get("site_composition") is None:
            alloy_sqs_kwargs["site_composition"] = site_composition

        if SQS_structure is None and any(
            alloy_sqs_kwargs.get(key, None) is None for key in self.sqs_required_kwargs
        ):
            raise ValueError(
                "You must specify either a predetermined SQS structure "
                f"or {self.sqs_required_kwargs} to generate the SQS structure!"
            )
        elif SQS_structure is None:
            coa = 1.0
            if alloy_sqs_kwargs["symmetry"] == "hcp":
                coa = (8.0 / 3.0) ** (0.5)

            sqs_job = alloy_mcsqs_job(
                lattice_abc={"a": 1.0, "coa": coa},
                **alloy_sqs_kwargs,
                anonymize_output=False,
            )
            SQS_structure = sqs_job.output["best sqs structure"]

        else:
            SQS_structure = SQS_structure.replace_species(
                {"X": list(site_composition)[0], "Y": list(site_composition)[1]}
            )

        v0 = sum(endpoint_volumes) / 2.0

        return Flow(jobs)
