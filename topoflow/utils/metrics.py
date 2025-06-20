"""This module defines various molecular metrics for evaluating chemical properties of molecules."""

import gzip
import math
import multiprocessing
import pickle
import time
import warnings

import numpy as np
import torch
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED
from torchmetrics import Metric

# Define a custom logP function using Mordred
_calc = Calculator(descriptors.SLogP, ignore_3D=True)
pool = None


def custom_logp(mol):
    """Calculates the water-octanol partition coefficient (LogP) of a molecule using Mordred.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        float: The calculated LogP value, or -3.0 if the molecule is invalid.
    """
    return _calc(mol)[0]


# Suppress RDKit warnings using RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _call_rdkit_func(func, mol):
    """Safely call an RDKit function on a molecule.

    Return None if a sanitization error (ValueError) occurs.
    """
    try:
        return func(mol)
    except ValueError:
        return None


def _worker_func(args):
    """The function we will call in parallel.

    Args:
        args (tuple): A tuple containing the molecule, the function to call, and a default value.
    """
    mol, func, default = args
    if mol is None:
        return default
    val = _call_rdkit_func(func, mol)
    return val if val is not None else default


def _normalize_score(scores, score_range):
    """Normalizes scores to be between 0 and 1."""
    return (scores - score_range[0]) / (score_range[1] - score_range[0])


def _raise_norm_warning(norm):
    """Raises a warning if normalization is specified but not used."""
    if norm is not None:
        warnings.warn(
            "Norm was specified but will not be used, as in MolGan implementation."
        )


def _constant_bump(x, x_low, x_high, decay=0.025):
    """Applies a bump function to values, primarily for normalization and scaling."""
    return np.select(
        condlist=[x <= x_low, x >= x_high],
        choicelist=[
            np.exp(-((x - x_low) ** 2) / decay),
            np.exp(-((x - x_high) ** 2) / decay),
        ],
        default=np.ones_like(x),
    )


def np_func(mol, np_model):
    """Calculates a score based on the likeness to natural products using a pre-trained model."""
    if mol is None:
        return -4  # Default score for invalid molecules

    # Get Morgan Fingerprint bits and their counts
    bits = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements()
    score = sum(np_model.get(bit, 0) for bit in bits) / float(mol.GetNumAtoms())

    # Apply logarithmic transformation for out-of-range scores
    if score > 4:
        score = 4 + math.log10(score - 4 + 1)
    elif score < -4:
        score = -4 - math.log10(-4 - score + 1)

    return score


class MolecularMetric(Metric):
    """Base class for molecular metrics."""

    def __init__(
        self,
        dataset,
        dist_sync_on_step=False,
        parallel=False,
        n_jobs=None,
    ):
        """Initializes the MolecularMetric.

        Implements basic functionality for metric calculation and aggregation.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.dataset = dataset
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.parallel = parallel
        if self.parallel:
            global pool
            self.n_jobs = n_jobs or multiprocessing.cpu_count()
            if pool is None:
                pool = multiprocessing.Pool(processes=self.n_jobs)

    @staticmethod
    def _valid_molecule(mol):
        """Checks if a molecule is valid and can be converted to a SMILES string."""
        return mol is not None and Chem.MolToSmiles(mol) != ""

    def update(self, mols):
        """Updates the metric's state with the provided molecules."""
        scores = self.compute_score(mols)
        scores = np.array(scores, dtype=np.float32)
        self.score += torch.tensor(scores).sum()
        self.total += len(mols)

    def compute(self):
        """Computes the final metric score."""
        return self.score.float() / self.total.float()

    def compute_score(self, mols):
        """Method to compute the score of the given molecules. Must be implemented by subclasses."""
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _map_scoring_func(self, func, default, mols):
        """Applies the scoring function to each molecule, either sequentially or in parallel."""
        if not self.parallel:
            # Sequential
            results = []
            for mol in mols:
                if mol is None:
                    results.append(default)
                else:
                    val = _call_rdkit_func(func, mol)
                    results.append(val if val is not None else default)
            return np.array(results, dtype=float)

        else:
            # Parallel
            # with multiprocessing.Pool(processes=self.n_jobs) as pool:
            args_iter = [(mol, func, default) for mol in mols]
            results = pool.map(_worker_func, args_iter)
            # results = pool.map(func, mols)
            return np.array(results, dtype=float)


class WOPCScore(MolecularMetric):
    """Calculates the water-octanol partition coefficient (LogP) of molecules."""

    _molgan_label = "logp"
    _molgan_func = "water_octanol_partition_coefficient"

    def __init__(
        self,
        dataset,
        norm=True,
        dist_sync_on_step=False,
        parallel=True,
        n_jobs=None,
    ):
        """Initializes the WOPCScore metric.

        Args:
            dataset (Dataset): The dataset containing molecules.
            norm (bool): Whether to normalize the scores. Defaults to True.
            dist_sync_on_step (bool): Whether to synchronize on step in distributed training. Defaults to False.
            parallel (bool): Whether to use parallel processing. Defaults to True.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to None.
        """
        super().__init__(
            dataset,
            dist_sync_on_step=dist_sync_on_step,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        self.norm = norm

    def compute_score(self, mols):
        """Computes the LogP score of molecules.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of LogP scores for each molecule, normalized between 0 and 1.
        """
        scores = self._map_scoring_func(custom_logp, -3.0, mols)
        if self.norm:
            scores = _normalize_score(scores, (-2.12178879609, 6.0429063424))
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class QEDScore(MolecularMetric):
    """Calculates the Quantitative Estimate of Drug-likeness (QED) of molecules."""

    _molgan_label = "qed"
    _molgan_func = "quantitative_estimation_druglikeness"

    def __init__(
        self,
        dataset,
        norm=None,
        dist_sync_on_step=False,
        parallel=True,
        n_jobs=None,
    ):
        """Initializes the QEDScore metric.

        Args:
            dataset (Dataset): The dataset containing molecules.
            norm (bool): Whether to normalize the scores. Defaults to True.
            dist_sync_on_step (bool): Whether to synchronize on step in distributed training. Defaults to False.
            parallel (bool): Whether to use parallel processing. Defaults to True.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to None.
        """
        super().__init__(
            dataset,
            dist_sync_on_step=dist_sync_on_step,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        _raise_norm_warning(norm)
        self.norm = norm

    def compute_score(self, mols):
        """Computes the QED score of molecules.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of QED scores for each molecule, normalized between 0 and 1.
        """
        scores = self._map_scoring_func(QED.qed, 0.0, mols)
        return scores


class NPScore(MolecularMetric):
    """Calculates the Natural Product (NP) score of molecules."""

    _molgan_label = "np"
    _molgan_func = "natural_product"
    np_model_path = "./data/NP_score.pkl.gz"

    def __init__(
        self,
        dataset,
        norm=True,
        dist_sync_on_step=False,
        parallel=False,
        n_jobs=None,
    ):
        """Initializes the NPScore metric.

        Args:
            dataset (Dataset): The dataset containing molecules.
            norm (bool): Whether to normalize the scores. Defaults to True.
            dist_sync_on_step (bool): Whether to synchronize on step in distributed training. Defaults to False.
            parallel (bool): Whether to use parallel processing. Defaults to True.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to None.
        """
        super().__init__(
            dataset,
            dist_sync_on_step=dist_sync_on_step,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        self.norm = norm
        self.np_model = pickle.load(gzip.open(self.np_model_path, "rb"))

    def compute_score(self, mols):
        """Computes the natural product score of molecules.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of natural product scores for each molecule, normalized between 0 and 1.
        """
        from functools import partial

        func = partial(np_func, np_model=self.np_model)

        scores = self._map_scoring_func(func, -4, mols)
        if self.norm:
            scores = _normalize_score(scores, (-3.0, 1.0))
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class SASScore(MolecularMetric):
    """Calculates the Synthetic Accessibility Score (SAS) of molecules."""

    _molgan_label = "sas"
    _molgan_func = "synthetic_accessibility"
    sa_model_path = "./data/SA_score.pkl.gz"

    def __init__(
        self,
        dataset,
        norm=True,
        dist_sync_on_step=False,
        parallel=True,
        n_jobs=None,
    ):
        """Initializes the SASScore metric.

        Args:
            dataset (Dataset): The dataset containing molecules.
            norm (bool): Whether to normalize the scores. Defaults to True.
            dist_sync_on_step (bool): Whether to synchronize on step in distributed training. Defaults to False.
            parallel (bool): Whether to use parallel processing. Defaults to True.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to None.
        """
        super().__init__(
            dataset,
            dist_sync_on_step=dist_sync_on_step,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        self.norm = norm
        loaded_sa = pickle.load(gzip.open(self.sa_model_path, "rb"))
        self.sa_model = {i[j]: float(i[0]) for i in loaded_sa for j in range(1, len(i))}

    def _compute_SAS(self, mol):
        """Computes the synthetic accessibility score for a given molecule.

        Args:
            mol (rdkit.Chem.Mol): The RDKit molecule object.

        Returns:
            float: The synthetic accessibility score, normalized between 1 and 10.
        """
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0

        for bitId, v in fps.items():
            nf += v
            score1 += self.sa_model.get(bitId, -4) * v
        score1 /= max(nf, 1e-9)  # avoid division by zero

        # Additional features
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        # Macrocycles
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = (
            0.0
            - sizePenalty
            - stereoPenalty
            - spiroPenalty
            - bridgePenalty
            - macrocyclePenalty
        )

        # Correction for fingerprint density
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5
        else:
            score3 = 0.0

        sascore = score1 + score2 + score3

        # Transform "raw" value into scale between 1 and 10
        _min = -4.0
        _max = 2.5
        sascore = 11.0 - (sascore - _min + 1) / (_max - _min) * 9.0

        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    def compute_score(self, mols):
        """Computes the synthetic accessibility score of molecules.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of synthetic accessibility scores for each molecule, normalized between 0 and 1.
        """
        scores = self._map_scoring_func(self._compute_SAS, 10.0, mols)
        scores = np.array(scores, dtype=float)
        if self.norm:
            scores = _normalize_score(scores, (5.0, 1.5))
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class NoveltyScore(MolecularMetric):
    """Calculates the novelty of molecules compared to a given dataset."""

    _molgan_label = "novelty"
    _molgan_func = "novel"

    def compute_score(self, mols):
        """Computes the novelty of molecules by checking if their SMILES representation is unique within the dataset.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of novelty scores for each molecule, normalized between 0 and 1.
        """

        def _score_func(mol):
            """Checks if a molecule's SMILES representation is novel compared to the dataset.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.

            Returns:
                float: 1.0 if the molecule is novel (not in dataset), otherwise 0.0.
            """
            s = Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""
            return float(s != "" and s not in self.dataset.smiles)

        scores = list(map(_score_func, mols))
        scores = np.array(scores, dtype=float)
        return scores


class DCScore(MolecularMetric):
    """Combines multiple molecular metrics to evaluate drug candidacy."""

    _molgan_label = "dc"
    _molgan_func = "drugcandidate"

    def compute_score(self, mols):
        """Computes a drug candidacy score based on multiple metrics.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of drug candidacy scores for each molecule, normalized between 0 and 1.
        """
        wopc = WOPCScore(
            self.dataset,
            norm=True,
            parallel=self.parallel,
            n_jobs=self.n_jobs,
        )(mols)
        sas = SASScore(
            self.dataset,
            norm=True,
            parallel=self.parallel,
            n_jobs=self.n_jobs,
        )(mols)
        novelty = NoveltyScore(
            self.data,
            parallel=self.parallel,
            n_jobs=self.n_job,
        )(mols)
        scores = 0.0
        scores += _constant_bump(wopc, 0.210, 0.945)
        scores += sas
        scores += novelty
        scores += (1 - novelty) * 0.3
        scores /= 4.0
        return scores


class UniqueScore(MolecularMetric):
    """Calculates the uniqueness of molecules within a batch."""

    _molgan_label = "unique"
    _molgan_func = "unique"

    def compute_score(self, mols):
        """Computes the uniqueness of molecules based on their SMILES representation.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of uniqueness scores for each molecule, normalized between 0 and 1.
        """
        smiles = [
            Chem.MolToSmiles(x, canonical=True) if x is not None else "" for x in mols
        ]
        # "Uniqueness" can be defined many ways; here we do 1 / frequency
        scores = 0.75 + np.array(
            [1 / smiles.count(s) if s != "" else 0 for s in smiles],
            dtype=np.float32,
        )
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class DiversityScore(MolecularMetric):
    """Calculates the chemical diversity of molecules within a batch relative to a dataset."""

    _molgan_label = "diversity"
    _molgan_func = "diversity"

    def compute_score(self, mols):
        """Computes the diversity of molecules based on Tanimoto similarity to a reference set.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of diversity scores for each molecule, normalized between 0 and 1.
        """

        def _compute_diversity(mol, fps):
            """Computes the diversity of a molecule based on Tanimoto similarity to a reference set.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.
                fps (list): List of reference Morgan fingerprints.

            Returns:
                float: The average Tanimoto distance of the molecule's fingerprint to the reference fingerprints.
            """
            ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 4, nBits=2048
            )
            dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
            return np.mean(dist)

        rand_mols = np.random.choice(self.dataset.mols, 100)
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            for mol in rand_mols
        ]

        def func(mol):
            """Wrapper function to compute diversity for a single molecule.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.

            Returns:
                float: The diversity score for the molecule, or 0.0 if the molecule is None.
            """
            return _compute_diversity(mol, fps) if mol is not None else 0.0

        scores = list(map(func, mols))
        scores = np.array(scores, dtype=float)
        scores = _normalize_score(scores, (0.9, 0.945))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class ValidityScore(MolecularMetric):
    """Checks the chemical validity of molecules (no '*' or '.' in SMILES, etc.)."""

    _molgan_label = "validity"
    _molgan_func = "valid"

    def compute_score(self, mols):
        """Computes the validity of molecules by checking their SMILES representation.

        Args:
            mols (list): List of RDKit molecule objects.

        Returns:
            np.ndarray: An array of scores where 1.0 indicates valid molecules and 0.0 indicates invalid ones.
        """

        def _compute_valid(mol):
            """Checks if a molecule is valid by ensuring its SMILES representation does not contain '*' or '.'.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.

            Returns:
                float: 1.0 if the molecule is valid (no '*' or '.' in SMILES), otherwise 0.0.
            """
            s = Chem.MolToSmiles(mol) if mol is not None else ""
            return float("*" not in s and "." not in s and s != "")

        scores = self._map_scoring_func(_compute_valid, 0.0, mols)
        return scores


ALL_METRICS = {
    "logp": WOPCScore,
    "qed": QEDScore,
    "np": NPScore,
    "sas": SASScore,
    "novelty": NoveltyScore,
    "dc": DCScore,
    "unique": UniqueScore,
    "diversity": DiversityScore,
    "validity": ValidityScore,
}


if __name__ == "__main__":
    smiles_list = [
        "CCO",
        "CCC",
        "CCN",
        "CCCl",
        "CCBr",
        "CCCC",
        "CC(C)C",
        "CCOCC",
        "CCOCCC",
        "CCN(CC)CC",
    ]  # * 10000
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Define metrics to benchmark
    metrics = [WOPCScore, QEDScore, NPScore, SASScore, NoveltyScore, DCScore]

    # Define the number of parallel jobs to benchmark
    job_counts = [1, 2, 4, 8, None]  # None uses all available CPUs

    for mclass in metrics:
        print(f"\nBenchmarking {mclass.__name__}")

        # Benchmark sequential execution
        sequential_metric = mclass(parallel=False)
        start_time = time.time()
        sequential_metric.update(mols)
        sequential_time = time.time() - start_time
        print(f"Sequential execution time: {sequential_time:.2f} seconds")
        sequential_scores = sequential_metric.compute_score(mols)
        continue
        # Benchmark parallel execution with different numbers of jobs
        for n_jobs in job_counts:
            print(f"\nTesting parallel execution with n_jobs={n_jobs}")
            parallel_metric = mclass(parallel=True, n_jobs=n_jobs)
            start_time = time.time()
            parallel_metric.update(mols)
            parallel_time = time.time() - start_time
            print(
                f"Parallel execution time with n_jobs={n_jobs}: {parallel_time:.2f} seconds"
            )

            # Compare results
            parallel_scores = parallel_metric.compute_score(mols)
            if np.allclose(parallel_scores, sequential_scores):
                print(
                    f"Results are identical between parallel (n_jobs={n_jobs}) and sequential execution."
                )
            else:
                print(
                    f"WARNING: Results differ for parallel (n_jobs={n_jobs}) and sequential execution."
                )

            # Print speedup factor
            if parallel_time < sequential_time:
                print(
                    f"Speedup factor with n_jobs={n_jobs}: {sequential_time / parallel_time:.2f}x (parallel faster)"
                )
            else:
                print(
                    f"Speedup factor with n_jobs={n_jobs}: {parallel_time / sequential_time:.2f}x (sequential slower)"
                )
