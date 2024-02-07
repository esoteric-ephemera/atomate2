from __future__ import annotations

import pytest
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.common.jobs import sqs


@pytest.mark.parametrize("symmetry", ["bcc", "fcc", "hcp", "zb", "ds"])
def test_alloy(symmetry):
    dict_key_to_type = {
        "primitive cell": Structure,
        "lattice_geom": dict,
        "symmetry": (str, int),
        "site composition": list,
    }
    for symm_name in sqs.Alloy._avail_symm[symmetry]:
        nbasis = 2 if symmetry in ["hcp", "zb", "ds"] else 1
        composition = [{"Al": 0.8, "Cu": 0.2} for _ in range(nbasis)]
        alloy = sqs.Alloy(
            lattice_abc={"a": 1.0, "c/a": (8.0 / 3.0) ** (0.5)},
            symmetry=symm_name,
            site_composition=composition,
        )
        assert alloy.symmetry == symmetry

        alloy_as_dict = alloy.as_dict()
        assert all(
            isinstance(alloy_as_dict[key], value)
            for key, value in dict_key_to_type.items()
        )


def test_alloy_sqs_maker(
    clean_dir,
    scaling: int = 2,
    instances: int = 2,
    lattice_abc: dict[str, float] | None = None,
    site_comp: dict[str, float] | None = None,
    symmetry: str = "hcp",
):
    common_kwargs = {
        "sqs_kwargs": {"sqs_method": "icet-enumeration"},
        "return_ranked_list": 1,
        "output_filename": None,
        "archive_instances": False,
        "anonymize_output": False,
    }

    lattice_abc = lattice_abc or {"a": 1.0, "c/a": (8.0 / 3.0) ** (0.5)}
    site_comp = site_comp or [{"Mg": 0.5, "Na": 0.5} for _ in range(2)]

    alloy = sqs.Alloy(
        lattice_abc=lattice_abc,
        symmetry=symmetry,
        site_composition=site_comp,
    )

    jobs = {}
    jobs["from structure"] = sqs.SqsMaker().make(
        structure=alloy.primitive_cell,
        scaling=scaling,
        instances=instances,
        **common_kwargs,
    )

    jobs["from alloy"] = sqs.AlloySqsMaker().make(
        symmetry=symmetry,
        site_composition=site_comp,
        instances=instances,
        scaling=scaling,
        lattice_abc=lattice_abc,
        **common_kwargs,
    )

    output = {}
    for job_kwarg in jobs:
        responses = run_locally(
            jobs[job_kwarg], create_folders=True, ensure_success=True
        )
        output[job_kwarg] = responses[jobs[job_kwarg].uuid][1].output
        assert output[job_kwarg]["input structure"] == alloy.primitive_cell
        assert isinstance(output[job_kwarg]["best sqs structure"], Structure)
        assert (
            output[job_kwarg]["best sqs structure"]
            == output[job_kwarg]["sqs structures"][0]["structure"]
        )
        assert isinstance(output[job_kwarg]["best objective"], float)

    assert output["from structure"] == output["from alloy"]

    common_kwargs["sqs_kwargs"].update(
        {
            "best_only": False,
            "remove_duplicate_structures": False,
        }
    )
    common_kwargs["return_ranked_list"] = True

    job = sqs.AlloySqsMaker().make(
        symmetry=symmetry,
        site_composition=site_comp,
        instances=instances,
        scaling=scaling,
        lattice_abc=lattice_abc,
        **common_kwargs,
    )

    responses = run_locally(job, create_folders=True, ensure_success=True)
    output = responses[job.uuid][1].output
    assert isinstance(output["sqs structures"], list)
    assert len(output["sqs structures"]) == instances
    assert isinstance(output["best objective"], float)
