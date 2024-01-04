"""Create disordered alloy structures with given symmetry."""

from __future__ import annotations

import os
import tarfile
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from monty.serialization import dumpfn
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.advanced_transformations import SQSTransformation
from pymatgen.transformations.standard_transformations import (
    ConventionalCellTransformation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from numpy.typing import ArrayLike
    from pymatgen.core import Element


class Alloy:
    """Generate an alloy structure with a specified symmetry and site composition."""

    _avail_symm: ClassVar[dict] = {
        "bcc": ("Im-3m", 229),
        "fcc": ("Fm-3m", 225),
        "hcp": ("P6_3/mmc", 194),
    }

    def __init__(
        self,
        a: float,
        symmetry: str | int,
        site_composition: dict[str | Element, float],
        c_over_a: float = 1.0,
    ) -> None:
        self.symmetry = None
        for symm_key in self._avail_symm:
            if symmetry == symm_key or symmetry in self._avail_symm[symm_key]:
                self.symmetry = symm_key
                break

        if self.symmetry is None:
            raise ValueError(f"Symmetry {symmetry} not yet implemented!")

        self.a = a
        self.c_over_a = c_over_a
        self.site_comp = site_composition

    def _direct_lattice_vectors(self) -> ArrayLike:
        dlv = {
            "bcc": [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
            "fcc": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            "hcp": [
                [0.5, -(3.0 ** (0.5)) / 2.0, 0.0],
                [0.5, 3.0 ** (0.5) / 2.0, 0.0],
                [0.0, 0.0, self.c_over_a],
            ],
        }
        return np.array(dlv[self.symmetry])

    def _basis_vectors(self) -> list:
        basis = {
            "bcc": [[0.0, 0.0, 0.0]],
            "fcc": [[0.0, 0.0, 0.0]],
            "hcp": [[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
        }
        return basis[self.symmetry]

    @property
    def primitive_cell(self) -> Structure:
        """Create a primitive disordered cell."""
        basis = self._basis_vectors()
        return Structure(
            lattice=Lattice(self.a * self._direct_lattice_vectors()),
            species=[self.site_comp for _ in range(len(basis))],
            coords=basis,
            coords_are_cartesian=False,
        )

    @property
    def conventional_cell(self) -> Structure:
        """Return the conventional cell for the disordered structure."""
        return ConventionalCellTransformation().apply_transformation(
            self.primitive_cell
        )

    def as_dict(self) -> dict:
        """Create JSON-able format of class."""
        return {
            "primitive cell": self.primitive_cell.as_dict(),
            "conventional cell": self.conventional_cell.as_dict(),
            "a": self.a,
            "c": self.c_over_a * self.a,
            "symmetry": self.symmetry,
            "site composition": self.site_comp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Alloy:
        """Create object from dict output of self.as_dict()."""
        return cls(
            a=d["a"],
            symmetry=d["symmetry"],
            site_composition=d["site composition"],
            c_over_a=d["c"] / d["a"],
        )


class MCSQS:
    """
    Generate Monte-Carlo SQS structures.

    Parameters
    ----------
    disordered_struct : Structure
        A pymatgen structure object that is disordered
    scaling : int | Sequence[int]
        Either the size of the supercell relative to disordered_struct, or the
        scaling of the supercell along each axis of disordered_struct
    sqs_kwargs : dict | None
        Options to pass to `SQSTransformation`
    """

    def __init__(
        self,
        disordered_struct: Structure,
        scaling: int | Sequence[int],
        sqs_kwargs: dict = None,
    ) -> None:
        if disordered_struct.is_ordered:
            raise ValueError("Your structure is likely ordered!")

        self.structure = disordered_struct

        if isinstance(scaling, int):
            nsites = scaling * len(self.structure)
        else:
            nsites = len(self.structure * scaling)

        num_sites = {}
        for element in self.structure.composition:
            num_sites[str(element)] = self.structure.composition[element] * nsites

        if not all(
            abs(num_sites[element] - round(num_sites[element])) < 1.0e-3
            for element in num_sites
        ):
            raise ValueError(
                f"Incompatible supercell number of sites {nsites} "
                f"for composition {self.structure.composition}"
            )

        self.scaling = scaling
        sqs_kwargs = sqs_kwargs or {}
        self.workdir = sqs_kwargs.get("directory", "./")
        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir, exist_ok=True)

        self.SQS = lambda nrun: SQSTransformation(
            scaling=scaling, instances=nrun, **sqs_kwargs
        )

    @classmethod
    def from_symm_comp(
        cls,
        a: float,
        symmetry: str | int,
        site_composition: dict[str | Element, float],
        sqs_scaling: int | Sequence[int],
        sqs_kwargs: dict = None,
        c_over_a: float = 1.0,
    ) -> MCSQS:
        """
        Instantiate class from Alloy input arguments.

        Parameters
        ----------
        lattice_abc : dict
            A dict of lattice parameters, either {"a": float},
            {"a": float, "c": float}, or {"a": float, "coa": float}, where coa = c / a
        symmetry : str | int
            Either the symmetry of the cell (e.g., fcc), the space group name
            (e.g., Fm-3m), or the space group number (e.g., 225)
        site_composition : dict
            A dict of elements and their relative composition on each site of the
            primitive disordered cell, e.g., {"Mg": 0.4, "Al": 0.6}
        nrun : int
            Number of parallel MCSQS runs to perform
        sqs_scaling : int | Sequence[int]
            Either the size of the supercell relative to disordered_struct, or the
            scaling of the supercell along each axis of disordered_struct
        sqs_kwargs : dict | None
            Options to pass to `SQSTransformation`
        return_ranked_list : bool | int
            Whether to return a list of SQS structures ranked by objective (bool), or
            how many to return ranked by objective (int)

        """
        alloy = Alloy(
            a=a, symmetry=symmetry, site_composition=site_composition, c_over_a=c_over_a
        )
        return cls(
            disordered_struct=alloy.primitive_cell,
            scaling=sqs_scaling,
            sqs_kwargs=sqs_kwargs,
        )

    def run_many(
        self,
        nrun: int,
        return_ranked_list: bool | int = False,
        output_filename: str | None = "MCSQS.json",
        archive_instances: bool = False,
    ) -> None:
        """
        Run parallel MCSQS instances for the same structure.

        Parameters
        ----------
        nrun : int
            Number of parallel instances to perform
        return_ranked_list: bool | int = False
            Whether to return a list of SQS structures ranked by objective function
            (bool), or how many to return (int). False returns only the best.
        output_filename : str | None
            If a str, the name of the file to log SQS output.
            If None, no file is written.

        Returns
        -------
        dict
            A dict of the best SQS structure, its objective (if saved), and
            the ranked SQS structures (if saved).
        """
        if return_ranked_list and nrun == 1:
            raise ValueError(
                "`return_ranked_list` should only be used for parallel MCSQS runs!"
                f" You set {nrun=} with {return_ranked_list=}."
            )

        sqs = self.SQS(nrun)
        self.sqs_structs = sqs.apply_transformation(
            self.structure, return_ranked_list=return_ranked_list
        )

        if return_ranked_list:
            self.best_sqs = self.sqs_structs[0]["structure"]
            self.best_objective = self.sqs_structs[0]["objective_function"]
        else:
            self.best_sqs = self.sqs_structs
            self.best_objective = None

            if os.path.isfile("bestcorr.out"):
                with open("bestcorr.out") as f:
                    best_corr_data = f.read()
                self.best_objective = best_corr_data.split("Objective_function=")[-1]
                self.best_objective = float(self.best_objective.strip())

        self.output = {
            "input structure": self.structure,
            "sqs structures": self.sqs_structs,
            "best sqs structure": self.best_sqs,
            "best objective": self.best_objective,
        }

        if output_filename:
            dumpfn(self.output, output_filename)


def alloy_mcsqs(
    lattice_abc: dict[str, float],
    symmetry: str | int,
    site_composition: dict[str | Element, float],
    nrun: int,
    sqs_scaling: int | Sequence[int],
    sqs_kwargs: dict = None,
    return_ranked_list: bool | int = False,
    archive_instances: bool = True,
) -> dict:
    """
    Make MCSQS alloy structures with atomate2.

    Parameters
    ----------
    lattice_abc : dict
        A dict of lattice parameters, either {"a": float},
        {"a": float, "c": float}, or {"a": float, "coa": float}, where coa = c / a
    symmetry : str | int
        Either the symmetry of the cell (e.g., fcc), the space group name
        (e.g., Fm-3m), or the space group number (e.g., 225)
    site_composition : dict
        A dict of elements and their relative composition on each site of the
        primitive disordered cell, e.g., {"Mg": 0.4, "Al": 0.6}
    nrun : int
        Number of parallel MCSQS runs to perform
    sqs_scaling : int | Sequence[int]
        Either the size of the supercell relative to disordered_struct, or the
        scaling of the supercell along each axis of disordered_struct
    sqs_kwargs : dict | None
        Options to pass to `SQSTransformation`
    return_ranked_list : bool | int
        Whether to return a list of SQS structures ranked by objective (bool), or
        how many to return ranked by objective (int)
    archive_instances : bool = False
        Whether to archive the contents of the SQS working directory as
        a tarball

    Returns
    -------
    dict
        A dict including the best SQS structure and its objective (if saved).
    """
    if lattice_abc.get("a", None) is None:
        raise ValueError("Need to specify lattice parameter a!")

    coa = 1.0
    if "c" in lattice_abc:
        coa = lattice_abc["c"] / lattice_abc["a"]
    elif "coa" in lattice_abc:
        coa = lattice_abc["coa"]

    default_sqs_kwargs = {
        "search_time": 60,
        "directory": "./instances/",
        "remove_duplicate_structures": True,
        "best_only": True,
    }

    sqs_kwargs = sqs_kwargs or {}
    default_sqs_kwargs.update(sqs_kwargs)

    original_directory = os.getcwd()

    mcsqs = MCSQS.from_symm_comp(
        a=lattice_abc["a"],
        symmetry=symmetry,
        site_composition=site_composition,
        sqs_scaling=sqs_scaling,
        sqs_kwargs=default_sqs_kwargs,
        c_over_a=coa,
    )

    mcsqs.run_many(
        nrun=nrun,
        return_ranked_list=return_ranked_list,
        output_filename="../MCSQS.json.gz",
    )

    os.chdir(original_directory)
    if archive_instances and isinstance(default_sqs_kwargs["directory"], str):
        archive_name: str = default_sqs_kwargs["directory"]
        if archive_name[-1] == "/":
            archive_name = archive_name[:-1]
        archive_name += ".tar.gz"

        # add files to tarball
        with tarfile.open(archive_name, "w:gz") as tarball:
            files = []
            for file in os.listdir(default_sqs_kwargs["directory"]):
                filename = os.path.join(default_sqs_kwargs["directory"], file)
                if os.path.isfile(filename):
                    files.append(filename)
                    tarball.add(filename)

        # cleanup
        for file in files:
            os.remove(file)

        if len(os.listdir(default_sqs_kwargs["directory"])) == 0:
            os.rmdir(default_sqs_kwargs["directory"])

    return mcsqs.output


alloy_mcsqs_job = job(alloy_mcsqs)
