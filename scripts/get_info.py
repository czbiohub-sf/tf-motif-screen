"""
This script extracts the MSA and template files from the CPU jobs after they are complete. Run in
conjuction with run_af3.py as a slurm job dependency.

Ben Iovino  3/6/25    CZ-Biohub
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import requests
import json
from af3tfm.dataset import DS, PROJECT_ROOT


def extract_cpu_data(job_dir: Path):
    """
    Extracts the data from the CPU jobs into individual files.

    Args:
        job_dir (str):
            Path to the directory of jobs.
    """

    for dir in Path(job_dir).iterdir():
        # For reasons unbeknownst to me, sometimes the json is nested one directory deeper
        try:
            with open(dir / f"{dir.name}/{dir.name}_data.json", "r") as f:
                data = json.load(f)
            with open(dir / f"{dir.name}_data.json", "w") as f:
                json.dump(data, f)

            # Remove the nested directory after extracting the json
            nested_dir = dir / dir.name
            for file in nested_dir.iterdir():
                file.unlink()
            nested_dir.rmdir()

        except FileNotFoundError:
            try:  # File is not nested
                with open(dir / f"{dir.name}_data.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                continue

        # Extract MSAs and write to separate .a3m files
        unpaired_msa = data["sequences"][0]["protein"]["unpairedMsa"]
        with open(dir / f"{dir.name}_unpaired.a3m", "w") as f:
            f.write(unpaired_msa)
        paired_msa = data["sequences"][0]["protein"]["pairedMsa"]
        with open(dir / f"{dir.name}_paired.a3m", "w") as f:
            f.write(paired_msa)

        # Extract structural templates and write to separate .mmcif files
        templates = data["sequences"][0]["protein"]["templates"]
        for i, template in enumerate(templates):
            with open(dir / f"{dir.name}_template_{i}.mmcif", "w") as f:
                f.write(template["mmcif"])


def get_msa_depth(tf_df: pd.DataFrame, msa_path: Path, df_file_path: str):
    """
    Saves the depth of the MSA for each TF in the TF dataframe.

    Args:
        tf_df (pd.DataFrame):
            Dataframe with TF information.
        msa_path (Path):
            Path to the MSAs.
        df_file_path (str):
            Path to the TF dataframe.
    """

    msa_depths = {}
    for tf in tf_df[["DBID"]].values:
        tf = tf[0]
        msa_file = f"{msa_path}/{tf.lower()}/{tf.lower()}_paired.a3m"

        # Count number of sequences
        msa_depth = 0
        with open(msa_file) as msa_file:
            for line in msa_file:
                if line[0] == ">":
                    msa_depth += 1
        msa_depths[tf] = msa_depth

    # Add MSA depths to TF dataframe
    tf_df["MSA_Depth"] = tf_df["DBID"].map(msa_depths)
    tf_df.to_csv(df_file_path, index=False)


def parse_mmcif(file_path):
    """
    Returns PDB ID from mmcif file if it contains a DNA linking annotation.

    Args:
        file_path (str):
            Path to the mmcif file.

    Returns:
        pdb_id (str):
            PDB ID.
    """

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                pdb_id = line.split("_")[1].strip()
            if "DNA linking" in line:
                return pdb_id

    return False


def get_pdb_templates(data_path: Path):
    """
    Writes PDB templates from HMMsearch to data/templates.

    Args:
        data_path (Path):
            Path to the directory of jobs.
    """

    # Get parent directory of jobs
    data_dir = str(data_path.parent)

    Path(f"{data_dir}/templates").mkdir(parents=True, exist_ok=True)
    for direc in Path(data_path).iterdir():
        tf = direc.parts[-1]
        Path(f"{data_dir}/templates/{tf}").mkdir(parents=True, exist_ok=True)

        for file in direc.glob("*.mmcif"):
            pdb_id = parse_mmcif(file)
            if pdb_id:
                request_pdb = requests.get(
                    f"https://files.rcsb.org/download/{pdb_id}.pdb1"
                )
                with open(f"{data_dir}/templates/{tf}/{pdb_id}.pdb", "wb") as f:
                    f.write(request_pdb.content)


def main():
    """ """

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--data_path", type=str, help="Path to the directory of jobs")
    args = parser.parse_args()

    dataset = DS[args.dataset]
    tf_df = pd.read_csv(dataset.seq_csv)
    data_path = Path(f"{PROJECT_ROOT}/{args.data_path}")

    extract_cpu_data(data_path)
    get_msa_depth(tf_df, data_path, dataset.seq_csv)
    get_pdb_templates(data_path)


if __name__ == "__main__":
    main()
