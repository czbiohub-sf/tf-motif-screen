"""
Retrieve the JASPAR database and convert it to a format compatible with HOMER.

Ben Iovino  8/13/25    Biohub
"""

import re
from pathlib import Path

import requests

from af3tfm.dataset import DATA_DIR


def download_jaspar(output_file: Path, redundant: bool = True):
    """
    Download the JASPAR database file and save it to the specified path.

    Args:
        output_file : Path
            The path to save the downloaded JASPAR file.
        redundant : bool, optional
            Whether to download the redundant version of the JASPAR database (default is True).
    """

    if redundant:
        url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_redundant_pfms_meme.txt"
    else:
        url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def generate_consensus(matrix: list[list[float]]) -> str:
    """
    Generates a consensus sequence from a position probability matrix.

    Args:
        matrix (list of list of float): The letter-probability matrix.

    Returns:
        str: The generated consensus sequence.
    """

    consensus, alphabet = "", "ACGT"
    for position_probs in matrix:
        max_prob_index = position_probs.index(max(position_probs))
        consensus += alphabet[max_prob_index]

    return consensus


def convert_meme_to_homer(input_file: Path, output_file: Path):
    """
    Reads a MEME file, converts motifs to Homer format, and writes to an output file.

    Args:
        input_file (Path):
            Path to the input file in MEME format.
        output_file (Path):
            Path to the output file for Homer format.
    """

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        current_motif_id = None
        current_motif_name = None
        matrix_width = 0
        matrix_lines_read = 0
        matrix = []

        for line in infile:
            line = line.strip()
            if not line:  # empty lines
                continue

            # Detect the start of a new motif
            if line.startswith("MOTIF"):
                if current_motif_id and matrix:  # Write previous motif if exists
                    consensus = generate_consensus(matrix)
                    outfile.write(f">{consensus}\t{current_motif_name}\t6.0\t0\t0\n")
                    for row in matrix:
                        formatted_row = "\t".join(f"{p:.6f}" for p in row)
                        outfile.write(f"{formatted_row}\n")
                    matrix = []  # Reset matrix for the new motif

                parts = line.split()
                current_motif_id = parts[1]
                current_motif_name = parts[1] if len(parts) > 2 else current_motif_id

            # Parse the letter-probability matrix line to get the width (w)
            elif line.startswith("letter-probability matrix"):
                match = re.search(r"w=\s*(\d+)", line)
                if match:
                    matrix_width = int(match.group(1))
                    matrix_lines_read = 0
                    matrix = []  # Reset matrix for the new motif data

            # Read the matrix data if we are inside a motif block
            elif (
                current_motif_id
                and matrix_width > 0
                and matrix_lines_read < matrix_width
            ):
                probs = [float(p) for p in line.split()]
                if len(probs) == 4:
                    matrix.append(probs)
                    matrix_lines_read += 1

        # Write the very last motif in the file
        if current_motif_id and matrix:
            consensus = generate_consensus(matrix)
            outfile.write(f">{consensus}\t{current_motif_name}\t6.0\t0\t0\n")
            for row in matrix:
                formatted_row = "\t".join(f"{p:.6f}" for p in row)
                outfile.write(f"{formatted_row}\n")


def main():
    """ """

    data_path = Path(f"{DATA_DIR}/tools/homer")
    jaspar_meme_file_path = Path(f"{data_path}/jaspar.meme.motif")
    jaspar_homer_file_path = Path(f"{data_path}/jaspar.homer.motif")

    download_jaspar(jaspar_meme_file_path)
    convert_meme_to_homer(jaspar_meme_file_path, jaspar_homer_file_path)


if __name__ == "__main__":
    main()
