"""
Runs NA-MPNN in specificity mode on the top-ranked AF3 predicted structures
for a given transcription factor, then builds a weighted-consensus PPM.

Ben Iovino 4/21/26  Biohub
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import prody

from af3tfm.dataset import DATA_DIR
from af3tfm.jobdb import Database

prody.confProDy(verbosity="none")

_NA_MPNN_DIR = Path(DATA_DIR) / "tools" / "NA-MPNN"
_INFERENCE_SCRIPT = _NA_MPNN_DIR / "inference" / "run.py"
_PYTHON = sys.executable


@dataclass
class Specificity:
    """
    Predicts a consensus position weight matrix (PWM) for a transcription
    factor using NA-MPNN specificity mode on AF3-predicted structures.

    Attributes:
        db (Database): Job database with structure paths and confidence scores.
        protein (str): Ensembl ID of the transcription factor.
        n_structs (int): Number of top structures to use for the consensus.
        n_batches (int): Number of NA-MPNN forward passes per structure.
        softmax_temp (float): Softmax temperature (>1 flattens weights toward uniform).
        out_subdir (str): Subdirectory of the job path where outputs are written.
        design_na_only (bool): If True, fix the protein chain(s) so NA-MPNN
            conditions DNA predictions on the true amino acid identities;
            otherwise protein and DNA are both masked and sampled.
    """

    db: Database
    protein: str
    n_structs: int = 50
    n_batches: int = 30  # Per hyperparameter sweep
    softmax_temp: float = 0.1  # Per hyperparameter sweep
    out_subdir: str = "specificities"
    design_na_only: bool = False

    def __post_init__(self):
        self.out_dir = (self.db.path / self.out_subdir).resolve()
        self._pdb_cache = self.out_dir / self.protein / "pdb_cache"
        self._npz_dir = self.out_dir / self.protein / "specificity"
        self._pdb_cache.mkdir(parents=True, exist_ok=True)
        self._npz_dir.mkdir(parents=True, exist_ok=True)

    def _get_top_structs(self) -> list[dict]:
        """
        Query the database for the top structures for self.protein, ranked by
        the product of ipTM and dom_pLDDT percentile ranks.

        Returns:
            structs (list[dict]): Dicts with keys: cif, dna, iptm, plddt, motif.
        """
        all_preds = pd.read_sql(
            """
            SELECT structure_path, dna, ipTM, dom_pLDDT, motif
            FROM runs
            WHERE protein = ?
            """,
            self.db.conn,
            params=(self.protein,),
        )
        all_preds["combined"] = all_preds["ipTM"].rank(pct=True) * all_preds[
            "dom_pLDDT"
        ].rank(pct=True)
        top = all_preds.nlargest(self.n_structs, "combined")

        return [
            {
                "cif": Path(r.structure_path),
                "dna": r.dna,
                "iptm": r.ipTM,
                "plddt": r.dom_pLDDT,
                "motif": r.motif,
            }
            for r in top.itertuples()
        ]

    def _cif_to_pdb(self, cif_path: Path, pdb_path: Path) -> None:
        """
        Convert an AF3 mmCIF file to PDB format.

        Args:
            cif_path (Path): Input mmCIF file.
            pdb_path (Path): Destination PDB file.
        """
        struct = prody.parseMMCIF(str(cif_path))
        prody.writePDB(str(pdb_path), struct)

    def _get_pdb(self, struct: dict) -> Path:
        """
        Return the PDB path for a structure, converting from CIF if not cached.

        Args:
            struct (dict): Structure dict from _get_top_structs().

        Returns:
            pdb_path (Path): Path to the PDB file.
        """
        pdb_path = (
            self._pdb_cache
            / f"{self.protein}_{struct['motif']}_{struct['dna']}_{struct['seed']}.pdb"
        )
        if not pdb_path.exists():
            self._cif_to_pdb(struct["cif"], pdb_path)

        return pdb_path

    def _run_na_mpnn(
        self, pdb_path: Path, out_folder: Path
    ) -> subprocess.CompletedProcess:
        """
        Run NA-MPNN in specificity mode on a single PDB structure.

        Args:
            pdb_path (Path): Input PDB file.
            out_folder (Path): NA-MPNN output directory (NPZ lands in out_folder/specificity/).

        Returns:
            CompletedProcess with stdout/stderr captured.
        """
        # Honor an explicit cap (NA_MPNN_N_THREADS) so SLURM tasks with a small
        # cpus-per-task don't oversubscribe the node; fall back to all cores.
        n_threads = os.environ.get("NA_MPNN_N_THREADS") or str(os.cpu_count() or 1)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = n_threads
        env["MKL_NUM_THREADS"] = n_threads
        cmd = [
            str(_PYTHON),
            str(_INFERENCE_SCRIPT),
            "--model_type",
            "na_mpnn",
            "--mode",
            "specificity",
            "--pdb_path",
            str(pdb_path),
            "--out_folder",
            str(out_folder),
            "--batch_size",
            "1",
            "--number_of_batches",
            str(self.n_batches),
            "--temperature",
            str(self.softmax_temp),
            "--output_pdbs",
            "0",
            "--output_specificity",
            "1",
            "--file_ending",
            "",
            "--design_na_only",
            str(int(self.design_na_only)),
        ]

        return subprocess.run(
            cmd, cwd=str(_NA_MPNN_DIR), capture_output=True, text=True, env=env
        )

    def _get_npz(self, struct: dict, pdb_path: Path) -> Path | None:
        """
        Return the NPZ path for a structure, running NA-MPNN if not cached.

        Args:
            struct (dict): Structure dict from _get_top_structs().
            pdb_path (Path): Path to the PDB file.

        Returns:
            npz_path (Path): Path to the NPZ file, or None if inference failed.
        """
        npz_path = (
            self._npz_dir
            / f"{self.protein}_{struct['motif']}_{struct['dna']}_{struct['seed']}.npz"
        )
        if not npz_path.exists():
            result = self._run_na_mpnn(pdb_path, self.out_dir / self.protein)
            if result.returncode != 0:
                print(f"NA-MPNN failed for {npz_path.name}:\n{result.stderr}")
                return None

        return npz_path

    def predict(self, motif: str, dna: str, seed: str) -> Path | None:
        """
        Run NA-MPNN specificity inference for a single structure.

        Args:
            motif (str): Motif identifier for the structure.
            dna (str): DNA sequence identifier for the structure.
            seed (str): Seed index (from the AF3 output folder name suffix).

        Returns:
            npz_path (Path): Path to the NPZ file, or None if inference failed.
        """
        row = pd.read_sql(
            # Escape "_" so seed "0" can't match "_10/" (bare "_" is a LIKE wildcard).
            "SELECT structure_path, ipTM, dom_pLDDT FROM runs WHERE protein = ? AND motif = ? AND dna = ? AND structure_path LIKE ? ESCAPE '\\'",
            self.db.conn,
            params=(self.protein, motif, dna, f"%\\_{seed}/%"),
        ).iloc[0]
        struct = {
            "cif": Path(row.structure_path),
            "dna": dna,
            "iptm": row.ipTM,
            "plddt": row.dom_pLDDT,
            "motif": motif,
            "seed": seed,
        }
        pdb_path = self._get_pdb(struct)
        npz_path = self._get_npz(struct, pdb_path)
        if npz_path is None or not npz_path.exists():
            print(f"  Skipping {self.protein}_{motif}_{dna} — no valid NPZ.")
            return None
        return npz_path
