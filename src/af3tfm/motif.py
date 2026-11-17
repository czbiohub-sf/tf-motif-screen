import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import motifs
from scipy.spatial import distance


@dataclass
class Motif:
    """
    Generate sequences from a position weight matrix (PWM).
    """

    name: str
    pwm_path: str = None
    motif_seqs: list[str] = None
    seq_names: list[str] = None
    seq_jsd: float = None
    bg_freq: dict[str, float] = None

    def __post_init__(self):
        # Load PWM from file (optional: sequences can also be injected directly)
        if self.pwm_path is not None:
            self.pwm = pd.read_csv(self.pwm_path, sep="\t", skiprows=1, header=None)
            if self.pwm.shape[1] == 5:
                self.pwm = self.pwm.iloc[:, 1:]
            self.pwm.columns = ["A", "C", "G", "T"]
            self.pwm = self.pwm.div(self.pwm.sum(axis=1), axis=0)

        # Initialize attributes (preserve any injected sequences)
        if self.motif_seqs is None:
            self.motif_seqs = []
        self.bg_freq = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}

    def _gen_seqs(self, n_seqs: int, nt_dist: pd.DataFrame) -> list[str]:
        """
        Nucleotide sequence generator.

        Args:
            n_seqs (int):
                Number of sequences to generate.
            nt_dist (pd.DataFrame):
                Nucleotide distribution to sample from.

        Returns:
            sequence (list[str]):
                List of generated sequences.
        """

        sequences = []
        for _ in range(n_seqs):
            sequence = []
            for j in range(self.pwm.shape[0]):
                seq = np.random.choice(nt_dist.columns, p=nt_dist.iloc[j].values)
                sequence.append(seq)
            sequences.append("".join(sequence))

        return sequences

    def generate_sequences(self, n_seqs: int, n_samples: int = 100):
        """
        Generates sequences for the TF based on the distribution provided.

        Args:
            n_seqs (int):
                Number of sequences to generate.
            n_samples (int):
                Number of samples to generate. Will take sample with best JSD.
        """

        best_score = 100
        nt_dist = self.pwm

        seqs = []
        for _ in range(n_samples):
            sequences = self._gen_seqs(n_seqs, nt_dist)

            # Check generated sequences against nucleotide distribution
            motif = motifs.create(sequences)
            pwm_gen = motif.counts.normalize(self.bg_freq)
            pwm_gen = pd.DataFrame(pwm_gen, columns=["A", "C", "G", "T"])
            js_divergence = distance.jensenshannon(
                self.pwm.values.flatten(), pwm_gen.values.flatten()
            )
            if js_divergence < best_score:
                seqs = sequences
                best_score = js_divergence

        self.seq_jsd = best_score
        self.motif_seqs = seqs


def load_homer_motif_file(path: str) -> tuple[str, np.ndarray]:
    """
    Parse a single HOMER .motif file.

    Args:
        path : str

    Returns:
        tuple[str, np.ndarray]
            Returns (motif_name, pwm) where pwm is (L, 4) float array [A, C, G, T].
    """

    with open(path) as f:
        lines = f.readlines()
    name = lines[0].split("\t")[1] if "\t" in lines[0] else lines[0].strip().lstrip(">")
    pwm = np.array(
        [list(map(float, line.split())) for line in lines[1:] if line.strip()],
        dtype=float,
    )

    return name, pwm


def load_homer_multi_motif_file(path: str) -> tuple[list[str], list[np.ndarray]]:
    """
    Parse a HOMER file containing multiple concatenated motifs.

    Args:
        path : str

    Returns:
        tuple[list[str], list[np.ndarray]]
            Returns (names, pwms) where each pwm is an (L, 4) float array [A, C, G, T].
    """

    names, pwms = [], []
    current_name, current_rows = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None and current_rows:
                    names.append(current_name)
                    pwms.append(np.array(current_rows, dtype=float))
                parts = line[1:].split("\t")
                current_name = parts[1] if len(parts) > 1 else parts[0]
                current_rows = []
            else:
                current_rows.append(list(map(float, line.split())))

    if current_name is not None and current_rows:
        names.append(current_name)
        pwms.append(np.array(current_rows, dtype=float))

    return names, pwms


def parse_pwm_file(file: Path) -> list[dict[str, str, np.ndarray]]:
    """
    Return a list of motifs from a HOMER PWM file.

    Args:
        file : Path
            Path to the PWM file in HOMER format.

    Returns:
        list[dict[str, str, np.ndarray]]: List of PWMs with their details
    """

    need_close = False
    if isinstance(file, (str, Path)):
        file = open(file, "r", encoding="utf-8")
        need_close = True

    pwms, current = {}, None
    for raw in file:
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:  # flush previous PWM
                current["matrix"] = np.asarray(current["matrix"], float)
                if current["name"] in pwms:
                    pwms[current["name"]] = (
                        pwms[current["name"]] + current["matrix"]
                    ) / 2  # Average current matrix
                else:
                    pwms[current["name"]] = [current["consensus"], current["matrix"]]
            header = line[1:]
            current = {
                "name": header.split()[1],
                "consensus": f"{header.split()[0]}",
                "matrix": [],
            }
        else:  # numeric row
            current["matrix"].append([float(x) for x in re.split(r"[ \t]+", line)])

    if current:  # last PWM
        current["matrix"] = np.asarray(current["matrix"], float)
        pwms[current["name"]] = [current["consensus"], current["matrix"]]

    if need_close:
        file.close()
    return pwms
