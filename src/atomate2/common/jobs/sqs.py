"""Create disordered alloy structures with given symmetry."""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from string import ascii_uppercase
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Maker, job
from monty.serialization import dumpfn
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.advanced_transformations import SQSTransformation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from numpy.typing import ArrayLike
    from pymatgen.core import Composition


def anonymizer(structure: Structure, starting_letter: int | str = 23) -> Structure:
    """
    Anonymize input structure composition.

    Parameters
    ----------
    structure : Structure
        Structure to anonymize composition
    starting_letter : int | str = 23
        Letter to start

    Returns
    -------
    Structure

    Ex: For a structure with formula Al3 Cu4 Ba2 Ca Dy10 Y,
    the output structure will have formula X3 Y4 Z2 A B10 C,
    where the substitution
        {"Al": "X", "Cu": "Y", "Ba": "Z", "Ca": "A", "Dy": "B", "Y": "C"}
    was performed.
    """
    species = structure.composition.elements
    letters = list(ascii_uppercase)

    if isinstance(starting_letter, str):
        starting_letter = letters.index(starting_letter)

    symbols = [
        letters[(i + starting_letter) % len(letters)] for i in range(len(letters))
    ]
    if len(species) > len(symbols):
        j = 0
        for i in range(len(species) - len(symbols)):
            new_char = symbols[i % len(letters)] + "_" + letters[j]
            symbols.append(new_char)
            if i % len(letters) == 0:
                j += 1

    replacement = {elt: symbols[i] for i, elt in enumerate(species)}
    return structure.replace_species(replacement)


class Alloy:
    """Generate an alloy structure with a specified symmetry and site composition."""

    _avail_symm: ClassVar[dict] = {
        "bcc": ("Im-3m", 229),
        "ds": ("Fd-3m", 227),
        "fcc": ("Fm-3m", 225),
        "zb": ("F-43m", 216),
        "hcp": ("P6_3/mmc", 194),
    }

    def __init__(
        self,
        lattice_abc: dict[str, float],
        symmetry: str | int,
        site_composition: list[Composition],
    ) -> None:
        """
        Generate an alloy structure from the list of self._avail_symm symmetries.

        Parameters
        ----------
        lattice_abc : dict[str,float]
            Must contain "a" as a kwarg. "b", "b/a", "c", "c/a" are optional.
        symmetry : str | int
            If a str, the name of the desired crystalline symmetry in
            self._avail_symm, or the space group. Else, the integer number
            of the desired space group.
        site_composition: list[Composition]
            A list of compositions (as dicts) on each site.
        """
        self.symmetry = self._get_symmetry(symmetry)

        if self.symmetry is None:
            raise ValueError(f"Symmetry {symmetry} not yet implemented!")

        self.lattice_geom = {"a": lattice_abc["a"], "b/a": 1.0, "c/a": 1.0}
        for key in ("b", "c"):
            if lattice_abc.get(key):
                self.lattice_geom[f"{key}/a"] = lattice_abc[key] / lattice_abc["a"]
            elif lattice_abc.get(f"{key}/a"):
                self.lattice_geom[f"{key}/a"] = lattice_abc[f"{key}/a"]

        self.site_comp = site_composition

    def _get_symmetry(self, symm: str | int) -> str:
        symmetry = None
        for symm_key in self._avail_symm:
            if symm == symm_key or symm in self._avail_symm[symm_key]:
                symmetry = symm_key
                break
        return symmetry

    def _direct_lattice_vectors(self) -> ArrayLike:
        dlv = {
            "bcc": [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
            "fcc": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            "hcp": [
                [0.5, -(3.0 ** (0.5)) / 2.0, 0.0],
                [0.5, 3.0 ** (0.5) / 2.0, 0.0],
                [0.0, 0.0, self.lattice_geom["c/a"]],
            ],
        }
        for key in ("zb", "ds"):
            dlv[key] = dlv["fcc"].copy()
        return np.array(dlv[self.symmetry])

    @staticmethod
    def _basis_vectors() -> dict:
        basis = {
            "bcc": [[0.0, 0.0, 0.0]],
            "fcc": [[0.0, 0.0, 0.0]],
            "ds": [[0.125, 0.125, 0.125], [0.875, 0.875, 0.875]],
            "hcp": [[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
        }
        basis["zb"] = basis["ds"].copy()
        return basis

    @property
    def primitive_cell(self) -> Structure:
        """Create a primitive disordered cell."""
        dlv = self._direct_lattice_vectors()
        basis = self._basis_vectors()[self.symmetry]
        return Structure(
            lattice=Lattice(self.lattice_geom["a"] * dlv),
            species=self.site_comp,
            coords=basis,
            coords_are_cartesian=False,
        )

    def as_dict(self) -> dict:
        """Create JSON-able format of class."""
        return {
            "primitive cell": self.primitive_cell,
            "lattice_geom": self.lattice_geom,
            "symmetry": self.symmetry,
            "site composition": self.site_comp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Alloy:
        """Create object from dict output of self.as_dict()."""
        return cls(
            lattice_abc=d["lattice_geom"],
            symmetry=d["symmetry"],
            site_composition=d["site composition"],
        )


class SQS:
    """
    Generate SQS structures using ATAT or icet.

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

    _defaults: ClassVar[dict] = {
        "search_time": 60,
        "directory": os.path.join(".", "instances"),
        "remove_duplicate_structures": True,
        "best_only": True,
    }

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
        self.sqs_kwargs = sqs_kwargs or {}

        if sqs_kwargs.get("sqs_method") is None:
            if nsites < 24:
                # For less than 24 atoms/cell, the exhaustive enumeration
                # method of icet always delivers the optimal SQS for a given
                # supercell size. Might be too expensive to run for more atoms/cell.
                self.sqs_kwargs["sqs_method"] = "icet-enumeration"
            else:
                # The ATAT Monte Carlo SQS method appears to be more robust for
                # larger supercells than the icet equivalent
                self.sqs_kwargs["sqs_method"] = "mcsqs"

        self._workdir = self.sqs_kwargs.get("directory", self._defaults["directory"])
        if not os.path.isdir(self._workdir):
            os.makedirs(self._workdir, exist_ok=True)

    def _sqs(self, instances: int) -> SQSTransformation:
        return SQSTransformation(
            scaling=self.scaling, instances=instances, **self.sqs_kwargs
        )

    @classmethod
    def from_symm_comp(
        cls,
        lattice_abc: dict[str, float],
        symmetry: str | int,
        site_composition: list[Composition],
        scaling: int | Sequence[int],
        sqs_kwargs: dict = None,
    ) -> SQS:
        """
        Instantiate SQS from an Alloy object.

        Parameters
        ----------
        lattice_abc : dict
            A dict of lattice parameters, either {"a": float},
            {"a": float, "c": float}, or {"a": float, "c/a": float}
        symmetry : str | int
            Either the symmetry of the cell (e.g., fcc), the space group name
            (e.g., Fm-3m), or the space group number (e.g., 225)
        site_composition : dict
            A dict of elements and their relative composition on each site of the
            primitive disordered cell, e.g., {"Mg": 0.4, "Al": 0.6}
        instances : int
            Number of parallel MCSQS runs to perform
        scaling : int | Sequence[int]
            Either the size of the supercell relative to disordered_struct, or the
            scaling of the supercell along each axis of disordered_struct
        sqs_kwargs : dict | None
            Options to pass to `SQSTransformation`
        """
        alloy = Alloy(
            lattice_abc=lattice_abc,
            symmetry=symmetry,
            site_composition=site_composition,
        )
        return cls(
            disordered_struct=alloy.primitive_cell,
            scaling=scaling,
            sqs_kwargs=sqs_kwargs,
        )

    def run_many(
        self,
        instances: int,
        return_ranked_list: bool | int = False,
        output_filename: str | None = "SQS.json.gz",
        anonymize_output: bool = False,
        archive_instances: bool = True,
    ) -> None:
        """
        Parallelized SQS search.

        For Monte Carlo methods, mcsqs and icet-monte_carlo, this
        executes parallel SQS searches from the same starting structure.

        For the icet-enumeration method, this divides the labor of
        searching through a list of structures.

        Parameters
        ----------
        instances : int
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
        original_directory = os.getcwd()

        if return_ranked_list and instances == 1:
            raise ValueError(
                "`return_ranked_list` should only be used for parallel MCSQS runs!"
                f" You set {instances=} with {return_ranked_list=}."
            )

        sqs = self._sqs(instances)
        self.sqs_structs = sqs.apply_transformation(
            self.structure, return_ranked_list=return_ranked_list
        )

        if return_ranked_list:
            self.best_sqs = self.sqs_structs[0]["structure"]
            self.best_objective = self.sqs_structs[0]["objective_function"]
        else:
            self.best_sqs = self.sqs_structs
            self.best_objective = None

            if self.sqs_kwargs["sqs_method"] == "mcsqs" and os.path.isfile(
                "bestcorr.out"
            ):
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

        if anonymize_output:
            non_list_keys = ["input structure", "best sqs structure"]
            if isinstance(self.output["sqs structures"], Structure):
                non_list_keys += ["sqs structures"]

            for key in non_list_keys:
                anonymizer(self.output[key])

            if "sqs structures" not in non_list_keys:
                for istruct in range(len(self.output["sqs structures"])):
                    anonymizer(self.output["sqs structures"][istruct]["structure"])

        os.chdir(original_directory)
        if output_filename:
            dumpfn(self.output, output_filename)

        if archive_instances and self.sqs_kwargs["sqs_method"] == "mcsqs":
            # MCSQS is the only SQS maker which requires a working directory
            archive_name = self.sqs_kwargs["directory"]
            if archive_name[-1] == os.path.sep:
                archive_name = archive_name[:-1]
            archive_name += ".tar.gz"

            # add files to tarball
            with tarfile.open(archive_name, "w:gz") as tarball:
                files = []
                for file in os.listdir(self.sqs_kwargs["directory"]):
                    filename = os.path.join(self.sqs_kwargs["directory"], file)
                    if os.path.isfile(filename):
                        files.append(filename)
                        tarball.add(filename)

            # cleanup
            for file in files:
                os.remove(file)

            if len(os.listdir(self.sqs_kwargs["directory"])) == 0:
                os.rmdir(self.sqs_kwargs["directory"])


@dataclass
class SqsMaker(Maker):
    """Make an SQS structure from an input disordered structure."""

    name: str = "SQS Maker"

    @job
    def make(
        self,
        structure: Structure,
        scaling: int | Sequence[int],
        instances: int,
        sqs_kwargs: dict | None = None,
        return_ranked_list: bool | int = 1,
        output_filename: str | None = "SQS.json.gz",
        anonymize_output: bool = False,
        archive_instances: bool = True,
    ) -> dict:
        """
        Run an SQS job.

        Parameters
        ----------
        structure: Structure
            Structure to perform an SQS on
        scaling : int | Sequence[int]
            Either the size of the supercell relative to disordered_struct, or the
            scaling of the supercell along each axis of disordered_struct
        instances : int
            Number of parallel SQS searches to run:
            - If enumeration is used,
            this divides the work of enumerating all possible SQS of a given size
            amongst workers
            - If Monte Carlo methods are used, this launches `instances` parallel
            searches
        sqs_kwargs : dict | None (default)
            Dict of kwargs to pass to SQSTransformation
        return_ranked_list : bool | int
            Whether to return a list of SQS structures ranked by objective (bool), or
            how many to return ranked by objective (int)
        output_filename : str | None, default = SQS.json.gz
            If a str, the name of the file to dump SQS results to
            (structures and corresponding objective function values).
        anonymize_output : bool = True
            Whether to anonymize the output (DummySpecies X, Y, Z, A, B,...)
        archive_instances : bool = False
            Whether to archive the contents of the SQS working directory as
            a tarball


        Returns
        -------
        dict
            A dict including the best SQS structure and its objective (if saved).
        """
        sqs = SQS(disordered_struct=structure, scaling=scaling, sqs_kwargs=sqs_kwargs)

        sqs.run_many(
            instances=instances,
            return_ranked_list=return_ranked_list,
            output_filename=output_filename,
            anonymize_output=anonymize_output,
            archive_instances=archive_instances,
        )

        return sqs.output


@dataclass
class AlloySqsMaker(Maker):
    """Make an SQS structure from an Alloy object."""

    name: str = "Alloy SQS maker"
    _default_lattice_params: dict = field(
        default_factory=lambda: {
            "hcp": {"a": 1.0, "b/a": 1.0, "c/a": (8.0 / 3.0) ** (0.5)}
        }
    )

    @job
    def make(
        self,
        symmetry: str | int,
        site_composition: list[Composition],
        instances: int,
        scaling: int | Sequence[int],
        lattice_abc: dict[str, float] | None = None,
        sqs_kwargs: dict | None = None,
        return_ranked_list: bool | int = 1,
        output_filename: str | None = "SQS.json.gz",
        archive_instances: bool = True,
        anonymize_output: bool = True,
    ) -> dict:
        """
        Make alloy structures with atomate2.

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
        instances : int
            Number of parallel MCSQS runs to perform
        scaling : int | Sequence[int]
            Either the size of the supercell relative to disordered_struct, or the
            scaling of the supercell along each axis of disordered_struct
        sqs_kwargs : dict | None (default)
            Dict of kwargs to pass to SQSTransformation
        return_ranked_list : bool | int
            Whether to return a list of SQS structures ranked by objective (bool), or
            how many to return ranked by objective (int)
        output_filename : str | None, default = SQS.json.gz
            If a str, the name of the file to dump SQS results to
            (structures and corresponding objective function values).
        anonymize_output : bool = True
            Whether to anonymize the output (DummySpecies X, Y, Z, A, B,...)
        archive_instances : bool = False
            Whether to archive the contents of the SQS working directory as
            a tarball

        Returns
        -------
        dict
            A dict including the best SQS structure and its objective (if saved).
        """
        default_lattice = {"a": 1.0, "b": 1.0, "c": 1.0}
        _symmetry = Alloy(default_lattice, symmetry, site_composition).symmetry
        default_lattice.update(self._default_lattice_params.get(_symmetry, {}))

        lattice_abc = lattice_abc or default_lattice

        sqs = SQS.from_symm_comp(
            lattice_abc=lattice_abc,
            symmetry=symmetry,
            site_composition=site_composition,
            scaling=scaling,
            sqs_kwargs=sqs_kwargs,
        )

        sqs.run_many(
            instances=instances,
            return_ranked_list=return_ranked_list,
            output_filename=output_filename,
            anonymize_output=anonymize_output,
            archive_instances=archive_instances,
        )

        return sqs.output
