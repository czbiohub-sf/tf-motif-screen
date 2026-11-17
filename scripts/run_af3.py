"""
Runs batch of AlphaFold3 jobs on SLURM.

Ben Iovino  12/13/24    CZ-Biohub
"""

import argparse
import copy
import subprocess as sp
from pathlib import Path

import pandas as pd
from af3tfm.config import CONFIG
from af3tfm.dataset import DS, PROJECT_ROOT

MAX_ARRAY_SIZE = 260000


def sort_job_names(job_names: list[str], tf_df: pd.DataFrame) -> list[str]:
    """
    Sorts job names by length of corresponding TF sequence.

    Args:
        job_names (list): List of job names.
        tf_df (pd.DataFrame): DataFrame containing transcription factor sequences.

    Returns:
        sorted_job_names (list): List of job names sorted by TF sequence length.
    """

    # Get TF lengths from sequences
    tf_df["SeqLength"] = tf_df["Sequence"].apply(len)
    tf_length_dict = dict(zip(tf_df["DBID"].str.lower(), tf_df["SeqLength"]))

    # Sort job names by length of corresponding TF sequence
    sorted_job_names = []
    for name in job_names:
        if (
            len(name.split("_")) > 2
        ):  # If motif suffix is present, remove it to access TF
            parts = name.split("_")
            dbid = "_".join(parts[:-2])
        else:
            dbid = name
        length = tf_length_dict.get(dbid.lower(), 0)  # 0 if not found
        sorted_job_names.append((length, name))

    sorted_job_names.sort(key=lambda x: x[0])  # Sort by length (first element of tuple)
    sorted_job_names = [name for length, name in sorted_job_names]

    return sorted_job_names


def write_job_names(
    tf_df: pd.DataFrame, input_path: Path, output_path: Path
) -> list[str]:
    """
    Writes job names to a text file for SLURM array jobs.

    Args:
        tf_df (pd.DataFrame): DataFrame containing transcription factor sequences.
        input_path (Path): The path to the directory of json input.
        output_path (Path): The path to the directory of AF3 output.

    Returns:
        job_names_sorted (list): List of job names sorted by TF sequence length.
    """

    # Check which jobs need to be run
    jobs_to_run = []
    for _, path in enumerate(input_path.iterdir()):
        # Search the whole job dir so timestamped subdirs are counted
        jobdir = output_path / path.name
        if (
            not path.is_dir()
            or any(jobdir.glob("*/*_model.cif"))
            or any(jobdir.rglob("*.a3m"))
        ):
            continue  # Skip if job already completed (final "*_model.cif", not per-seed cifs)
        jobs_to_run.append(path.name)

    try:
        jobs_to_run.remove("slurm_logs")
    except ValueError:
        print('No "slurm_logs" directory to remove from job list.')
    print(len(jobs_to_run), "jobs to run.")

    job_names_sorted = sort_job_names(jobs_to_run, tf_df)

    # Write all jobs to a single file for reference
    with open(f"{input_path}/job_names.txt", "w") as f:
        for job_name in job_names_sorted:
            f.write(f"{job_name}\n")

    return job_names_sorted


def write_job_chunks(job_names: list[str], input_path: Path) -> list[tuple[str, int]]:
    """
    Splits job names into chunks and writes each chunk to a separate file.

    Args:
        job_names (list): List of all job names.
        input_path (Path): The path to the directory of json input.

    Returns:
        list[tuple[str, int]]: List of tuples containing (chunk_file_path, num_jobs_in_chunk).
    """

    chunks = []
    num_chunks = (len(job_names) + MAX_ARRAY_SIZE - 1) // MAX_ARRAY_SIZE

    for i in range(num_chunks):
        start_idx = i * MAX_ARRAY_SIZE
        end_idx = min((i + 1) * MAX_ARRAY_SIZE, len(job_names))
        chunk = job_names[start_idx:end_idx]

        chunk_file = f"{input_path}/job_names_chunk_{i+1}.txt"
        with open(chunk_file, "w") as f:
            for job_name in chunk:
                f.write(f"{job_name}\n")
        chunks.append((chunk_file, len(chunk)))

    return chunks


def get_config(**kwargs) -> dict:
    """
    Return a copy of the af3tfm configuration (config.yaml + config.local.yaml).

    Args:
        **kwargs: The arguments to update the configuration dictionary.

    Returns:
        dict: The configuration dictionary.
    """

    config = copy.deepcopy(CONFIG)

    for key, value in kwargs.items():
        config[key] = value

    return config


def cpu_config(input_path: str, output_path: str, **kwargs) -> dict:
    """
    Returns config for a CPU job.

    Args:
        path (str): The path to the JSON file (containing protein sequence).
        output_path (str): The path to the output directory.
        **kwargs: Additional arguments to update the configuration dictionary.

    Returns:
        dict: The configuration dictionary.
    """

    config = get_config(
        time="24:00:00",
        mem="128G",
        gpus=0,
        cpus_per_task=8,
        input_dir=input_path,
        output_dir=output_path,
        partition=kwargs["partition"],
        run_inference=False,
    )

    # Optional node exclusion
    if kwargs.get("exclude") is not None:
        config["exclude"] = kwargs["exclude"]

    return config


def gpu_config(input_path: str, output_path: str, **kwargs) -> dict:
    """
    Returns config for a GPU job.

    Args:
        input_path (str): The path to the JSON file (containing protein and DNA ssequences)
        output_path (str): The path to the output directory.
        **kwargs: Additional arguments to update the configuration dictionary.

    Returns:
        dict: The configuration dictionary.
    """

    config = get_config(
        mem="16G",
        gpus=1,
        cpus_per_task=1,
        input_dir=input_path,
        output_dir=output_path,
        json_file="_data.json",
        partition=kwargs["partition"],
        run_data_pipeline=False,
    )

    # Specific GPU options
    if kwargs["constraint"] is not None:
        config["constraint"] = kwargs["constraint"]

    # Optional node exclusion
    if kwargs.get("exclude") is not None:
        config["exclude"] = kwargs["exclude"]

    return config


def generate_slurm_script(config: dict[str], num_jobs: int, job_list_file: str) -> str:
    """
    Generate a slurm script for AlphaFold3.

    Args:
        config (dict): The configuration dictionary.
        num_jobs (int): The number of jobs to run.
        job_list_file (str): Path to the file containing job names for this chunk.

    Returns:
        str: The slurm script.
    """

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=alphafold3
#SBATCH --time={config["time"]}
#SBATCH --mem={config["mem"]}
#SBATCH --partition={config["partition"]}
#SBATCH --ntasks={config["ntasks"]}
#SBATCH --cpus-per-task={config["cpus_per_task"]}
#SBATCH --mail-type={config["mail_type"]}
#SBATCH --output={config["input_dir"]}/slurm_logs/%j.log
#SBATCH --error={config["input_dir"]}/slurm_logs/%j.err
#SBATCH --array=1-{num_jobs}
"""

    # Optional mail recipient (unset -> SLURM's default for the submitting user)
    if config["mail_user"]:
        slurm_script += f"#SBATCH --mail-user={config['mail_user']}\n"

    # Optional node exclusion
    if config.get("exclude", "None") != "None":
        slurm_script += f"#SBATCH --exclude={config['exclude']}\n"

    # GPU specific options
    if config["partition"] == "gpu" or config["partition"] == "preempted":
        slurm_script += f"#SBATCH --gpus={config['gpus']}\n"
    if config["constraint"] != "None":
        slurm_script += f"#SBATCH --constraint={config['constraint']}\n"

        slurm_script += """
"""

    slurm_script += f"""
JOB_ID_LIST="{job_list_file}"
CURRENT_JOB_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$JOB_ID_LIST")

OUT_PARENT="{config["output_dir"]}/${{CURRENT_JOB_ID}}"
if find "$OUT_PARENT" -type f -name '*_model.cif' -print -quit | grep -q .
then
    echo "[${{SLURM_JOB_ID}}]  Results already exist in ${{OUT_PARENT}}; skipping."
    exit 0
fi
"""

    slurm_script += f"""
module load alphafold/3.0.1-23-g792e61e
alphafold \
--json_path={config["input_dir"]}/"$CURRENT_JOB_ID"/"${{CURRENT_JOB_ID}}{config["json_file"]}" \
--output_dir={config["output_dir"]}/"${{CURRENT_JOB_ID}}" \
"""

    # Optional AF3 flags to separate steps
    if config["run_inference"] is False:
        slurm_script += "--run_inference=false"
    if config["run_data_pipeline"] is False:
        slurm_script += "--run_data_pipeline=false"

    return slurm_script


def generate_cpu_dependency_script(dataset: str, data_path: str) -> str:
    """
    Generates a slurm script to run get_info.py after all CPU jobs are complete.

    Args:
        dataset (str): The dataset to use.
        data_path (str): The path to the directory of CPU output.

    Returns:
        str: The slurm script.
    """

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=extract_cpu_data
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END

module load anaconda
module load data.science
conda activate af3-tfm

python {PROJECT_ROOT}/scripts/get_info.py --dataset={dataset} --data_path={data_path}
"""

    return slurm_script


def get_job_config(
    job_type: str,
    input_path: str,
    output_path: str,
    partition: str,
    constraint: str = None,
    exclude: str = None,
) -> dict:
    """
    Returns the appropriate configuration based on job type.

    Args:
        job_type (str): Type of job (cpu or gpu).
        input_path (str): Path to input directory.
        output_path (str): Path to output directory.
        partition (str): SLURM partition.
        constraint (str): SLURM constraint (GPU only).
        exclude (str): Comma-separated nodes to exclude (ex. gpu-i-1).

    Returns:
        dict: The configuration dictionary.
    """

    if job_type == "cpu":
        return cpu_config(input_path, output_path, partition=partition, exclude=exclude)
    elif job_type == "gpu":
        return gpu_config(
            input_path,
            output_path,
            partition=partition,
            constraint=constraint,
            exclude=exclude,
        )
    else:
        raise ValueError(f"Invalid job_type: {job_type}. Must be 'cpu' or 'gpu'.")


def submit_chunked_jobs(
    config: dict, job_names: list[str], input_path: Path, job_type: str
) -> None:
    """
    Submits multiple chunked array jobs for large job sets.

    Args:
        config (dict): The configuration dictionary.
        job_names (list): List of all job names.
        input_path (Path): Path to input directory.
        job_type (str): Type of job (cpu or gpu).
    """

    job_chunks = write_job_chunks(job_names, input_path)
    print(
        f"\nSubmitting {len(job_chunks)} array job(s) for {len(job_names)} total {job_type.upper()} jobs..."
    )

    for chunk_idx, (chunk_file, num_jobs_in_chunk) in enumerate(job_chunks, start=1):
        slurm_script = generate_slurm_script(config, num_jobs_in_chunk, chunk_file)
        script_file = f"{input_path}/run_af3_{job_type}_chunk_{chunk_idx}.sh"

        with open(script_file, "w") as f:
            f.write(slurm_script)

        slurm_result = sp.run(
            ["sbatch", "--parsable", script_file],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = slurm_result.stdout.strip()
        print(
            f"Chunk {chunk_idx}: Submitted SLURM array job with ID {job_id} ({num_jobs_in_chunk} jobs)"
        )


def submit_single_job(
    config: dict,
    job_names: list[str],
    input_path: Path,
    output_path: str,
    job_type: str,
    dataset_name: str = None,
) -> None:
    """
    Submits a single array job and handles CPU dependency job if needed.

    Args:
        config (dict): The configuration dictionary.
        job_names (list): List of all job names.
        input_path (Path): Path to input directory.
        output_path (str): Path to output directory.
        job_type (str): Type of job (cpu or gpu).
        dataset_name (str): Dataset name (required for CPU jobs).
    """

    slurm_script = generate_slurm_script(
        config, len(job_names), f"{input_path}/job_names.txt"
    )
    script_file = f"{input_path}/run_af3_{job_type}.sh"

    with open(script_file, "w") as f:
        f.write(slurm_script)

    slurm_result = sp.run(
        ["sbatch", "--parsable", script_file],
        capture_output=True,
        text=True,
        check=True,
    )

    job_id = slurm_result.stdout.strip()
    print(f"Submitted SLURM array job with ID {job_id} ({len(job_names)} jobs)")

    # Submit dependency job for CPU jobs
    if job_type == "cpu":
        dependency_script = generate_cpu_dependency_script(dataset_name, output_path)
        sp.run(
            ["sbatch", f"--dependency=afterany:{job_id}"],
            input=dependency_script,
            text=True,
            check=True,
        )
        print("Dependency job submitted successfully.")


def main():
    """
    Partitions: cpu, gpu, preempted
    Constraints: [h100|a100_80|h200] or None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, help="Path to the directory of json input"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use.")
    parser.add_argument("--job_type", type=str, help="Type of job (cpu or gpu).")
    parser.add_argument("--partition", type=str, help="Slurm partition.")
    parser.add_argument("--constraint", type=str, help="Slurm constraint (GPU only).")
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated nodes to exclude (ex. gpu-i-1).",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to the directory of AF3 output."
    )
    args = parser.parse_args()

    # Load dataset and prepare paths
    dataset = DS[args.dataset]
    tf_df = pd.read_csv(dataset.seq_csv)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_path)

    # Generate job list
    job_names = write_job_names(tf_df, input_path, output_path)
    if not job_names:
        print("No jobs to submit.")
        return

    config = get_job_config(
        args.job_type,
        args.input_path,
        args.output_path,
        args.partition,
        args.constraint,
        args.exclude,
    )

    if args.job_type == "gpu" and len(job_names) > MAX_ARRAY_SIZE:
        submit_chunked_jobs(config, job_names, input_path, args.job_type)
    else:
        submit_single_job(
            config, job_names, input_path, args.output_path, args.job_type, dataset.name
        )


if __name__ == "__main__":
    main()
