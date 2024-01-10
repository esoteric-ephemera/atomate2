import pytest
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.common.jobs import alloy_sqs


@pytest.mark.parametrize("symmetry", ["bcc", "fcc", "hcp"])
def test_alloy(symmetry):
    expected_symmetry = {
        "bcc": ("bcc", "Im-3m", 229),
        "fcc": ("fcc", "Fm-3m", 225),
        "hcp": ("hcp", "P6_3/mmc", 194),
    }
    for symm_name in expected_symmetry[symmetry]:
        alloy = alloy_sqs.Alloy(
            a=1.0,
            symmetry=symm_name,
            site_composition={"Al": 0.8, "Cu": 0.2},
            c_over_a=1.0,
        )
        assert alloy.symmetry == symmetry


def test_alloy_mcsqs(clean_dir, runs: int = 2):
    lattice_abc = {"a": 1.0, "coa": 1.5}
    symmetry = "hcp"
    comp = {"Mg": 0.5, "Na": 0.5}

    alloy = alloy_sqs.Alloy(
        a=lattice_abc["a"],
        symmetry=symmetry,
        site_composition=comp,
        c_over_a=lattice_abc["coa"],
    )

    job = alloy_sqs.alloy_mcsqs_job(
        lattice_abc=lattice_abc,
        symmetry=symmetry,
        site_composition=comp,
        nrun=1,
        sqs_scaling=2,
        sqs_kwargs={"search_time": 0.01},
        return_ranked_list=False,
        anonymize_output=False,
    )
    responses = run_locally(job, create_folders=True, ensure_success=True)
    output = responses[job.uuid][1].output

    assert output["input structure"] == alloy.primitive_cell
    assert isinstance(output["best sqs structure"], Structure)
    assert output["best sqs structure"] == output["sqs structures"]
    assert isinstance(output["best objective"], float)

    sqs_kwargs = {
        "search_time": 0.01,
        "best_only": False,
        "remove_duplicate_structures": False,
    }

    job = alloy_sqs.alloy_mcsqs_job(
        lattice_abc=lattice_abc,
        symmetry=symmetry,
        site_composition=comp,
        nrun=runs,
        sqs_scaling=2,
        sqs_kwargs=sqs_kwargs,
        return_ranked_list=True,
    )
    responses = run_locally(job, create_folders=True, ensure_success=True)
    output = responses[job.uuid][1].output
    assert isinstance(output["sqs structures"], list)
    assert len(output["sqs structures"]) == runs
    assert isinstance(output["best objective"], float)
