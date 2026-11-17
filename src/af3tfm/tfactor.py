from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json
from af3tfm.dataset import PROJECT_ROOT
from af3tfm.motif import Motif
from af3tfm.jobdb import Database
import string


@dataclass
class TFactor:
    """
    Stores information about a transcription factor for AF3.
    """

    name: str
    ensembl_ID: str
    sequence: str
    motif: Motif = None
    data_path: str = None
    db: Database = None

    def _write_job(self, job_data: dict, direc: str, job_name: str, job_type: str):
        """
        Writes a json input file for AF3.

        Args:
            job_data (dict):
                Job data to write.
            direc (str):
                Directory to save the job data.
            job_name (str):
                Name of the job.
            job_type (str):
                Type of job (cpu, gpu).
        """

        if job_type == "cpu":  # input does not contain MSA, no _data suffix
            input_file_path = f"{direc}/{job_name}/{job_name}.json"
        if job_type == "gpu":  # input contains MSA, need _data suffix
            input_file_path = f"{direc}/{job_name}/{job_name}_data.json"

        Path(f"{direc}/{job_name}").mkdir(parents=True, exist_ok=True)
        if Path(
            input_file_path
        ).exists():  # Don't want to overwrite already existing files
            return

        with open(input_file_path, "w") as f:
            json.dump(job_data, f, indent=4)

    def _cpu_job(self, direc: str):
        """
        Creates a job for the CPU.

        Args:
            direc: str
                Directory to save the job data.
        """

        job_data = {
            "name": self.ensembl_ID.lower(),
            "modelSeeds": [1],
            "sequences": [],
            "dialect": "alphafold3",
            "version": 1,
        }

        # Add protein to the job
        job_data["sequences"].append(
            {
                "protein": {
                    "sequence": self.sequence,
                    "id": "A",
                },
            }
        )

        self._write_job(job_data, direc, self.ensembl_ID.lower(), "cpu")

    def _gpu_job(self, direc: str):
        """
        Creates one GPU job for each motif stored in self.motif_seqs.
        Assumes CPU step has been run and sets MSA file path as input.

        Args:
            direc (str):
                Directory to save the job data.
        """

        alphabet = string.ascii_uppercase

        # All data files should be in same place
        # ex. {PROJECT_ROOT}/data/jobs/{motif-nonmotif}/output_fold/{ensembl_ID}
        data_file_path = f"{PROJECT_ROOT}/{self.data_path}/{self.ensembl_ID.lower()}/{self.ensembl_ID.lower()}_data.json"
        unpairedMsa_file_path = f"{PROJECT_ROOT}/{self.data_path}/{self.ensembl_ID.lower()}/{self.ensembl_ID.lower()}_unpaired.a3m"
        pairedMsa_file_path = f"{PROJECT_ROOT}/{self.data_path}/{self.ensembl_ID.lower()}/{self.ensembl_ID.lower()}_paired.a3m"

        # Already exists from _cpu_job
        with open(data_file_path, "r") as f:
            data = json.load(f)
        protein = data["sequences"][0]["protein"]

        # Define templates (correspond file paths to indices from data.json)
        templates = []
        for i, template in enumerate(protein["templates"]):
            template_file_path = str(
                f"{PROJECT_ROOT}/{self.data_path}/{self.ensembl_ID.lower()}/{self.ensembl_ID.lower()}_template_{i}.mmcif"
            )
            templates.append(
                {
                    "mmcifPath": f"{template_file_path}",
                    "queryIndices": template["queryIndices"],
                    "templateIndices": template["templateIndices"],
                }
            )

        # Create a job for each motif
        for i, seq in enumerate(self.motif.motif_seqs):
            if self.db is not None:  # Skip if sequence already exists in database
                self.db.cursor.execute(
                    "SELECT 1 FROM runs WHERE dna = ? AND protein = ? LIMIT 1",
                    (seq, self.ensembl_ID.lower()),
                )
                if self.db.cursor.fetchone() is not None:
                    continue

            rev_comp = (
                seq.replace("A", "t")
                .replace("T", "a")
                .replace("C", "g")
                .replace("G", "c")
                .upper()[::-1]
            )

            job_data = {
                "name": self.ensembl_ID.lower(),
                "modelSeeds": [1, 2, 3, 4, 5],
                "sequences": [],
                "dialect": "alphafold3",
                "version": 1,
            }

            job_data["sequences"].append(
                {
                    "protein": {
                        "sequence": protein["sequence"],
                        "id": protein["id"],
                        "modifications": [],
                        "unpairedMsaPath": f"{unpairedMsa_file_path}",
                        "pairedMsaPath": f"{pairedMsa_file_path}",
                        "templates": templates,
                    },
                }
            )
            job_data["sequences"].append({"dna": {"sequence": seq, "id": alphabet[1]}})
            job_data["sequences"].append(
                {"dna": {"sequence": rev_comp, "id": alphabet[2]}}
            )

            # Name for json input file
            if self.motif.seq_names is not None:
                # Per-sequence names (already unique, e.g. genomic negative IDs)
                job_name = f"{self.ensembl_ID.lower()}_{self.motif.seq_names[i]}"
            elif self.motif.name:
                job_name = f"{self.ensembl_ID.lower()}_{self.motif.name}_{i}"
            else:
                job_name = f"{self.ensembl_ID.lower()}_{i}"

            self._write_job(job_data, direc, job_name, "gpu")

    def create_af_job(self, job_type: str, direc: str):
        """
        Writes a json input file for AF3.

        Args:
            job_type (str):
                Type of job to create (cpu, gpu).
            direc (str):
                Directory to save the job data.
        """

        if job_type == "cpu":
            self._cpu_job(direc)
        if job_type == "gpu":
            self._gpu_job(direc)


def load_tfs(
    tf_df: pd.DataFrame, name: str, data_path: str = None, database: Database = None
) -> dict[str, TFactor]:
    """
    Return a dictionary of TFactor objects from a dataframe containing information about TFs.

    Args:
        tf_df (pd.DataFrame):
            A dataframe containing information about TFs.
        name (str):
            Label dict key as "TF_Name" or "DBID".
        data_path (str):
            Path to the directory of data files (optional).
        database (Database):
            Database object to check for existing sequences (optional).

    Returns:
        tfs (dict[str, TFactor]):
            A dictionary of TFactor objects.
    """

    tfs: dict[str, TFactor] = {}
    for _, row in tf_df.iterrows():
        tf = TFactor(
            name=row["TF_Name"],
            ensembl_ID=row["DBID"],
            sequence=row["Sequence"],
            db=database,
        )

        if data_path:
            tf.data_path = data_path

        # Store the TF object in the dictionary
        if name == "TF_Name":
            tfs[row["TF_Name"]] = tf
        if name == "DBID":
            tfs[row["DBID"]] = tf

    return tfs
