"""
Downloads protein sequences from ensembl.

Ben Iovino  5/02/25    CZ-Biohub
"""

import time

import pandas as pd
import requests
from requests.exceptions import RequestException


def _get_json(url, *, params=None, retries=4, timeout=20):
    """GET a JSON endpoint, retrying transient failures with backoff.

    Returns the parsed JSON, or None if the request keeps failing / is non-200.
    Retries protect the curated TF set from being silently dropped by a flaky
    Ensembl connection (the caller drops any TF whose sequence comes back None).
    """
    headers = {"Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=timeout
            )
        except RequestException as e:  # Random errors may occur
            print(f"Request failed ({attempt + 1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:  # rate limited; honor Retry-After
            time.sleep(float(response.headers.get("Retry-After", 2)))
            continue
        return None
    return None


def fetch_canonical_protein(gene_id) -> str:
    """
    Fetch the canonical protein sequence for a given Ensembl gene ID.

    Args:
        gene_id (str):
            Ensembl gene ID (ENSG*).

    Returns:
        str:
            Canonical protein sequence in FASTA format.
    """

    # Step 1: Get the canonical transcript
    data = _get_json(f"https://rest.ensembl.org/lookup/id/{gene_id}")
    if data is None:
        return None

    canonical_transcript = data.get("canonical_transcript")
    canonical_transcript = canonical_transcript.split(".")[0]

    if not canonical_transcript:
        raise Exception(f"No canonical transcript found for gene ID: {gene_id}")

    # Step 2: Get the protein sequence for the canonical transcript
    seq_data = _get_json(
        f"https://rest.ensembl.org/sequence/id/{canonical_transcript}",
        params={"type": "protein"},
    )
    if seq_data is None:
        print(f"Failed to fetch sequence for {gene_id}")
        return None

    print(f"Fetched sequence for {gene_id}")
    return seq_data["seq"]


def _post_json(url, payload, *, params=None, retries=4, timeout=60):
    """POST to a JSON endpoint, retrying transient failures with backoff.

    Returns the parsed JSON, or None if the request keeps failing / is non-200.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(retries):
        try:
            response = requests.post(
                url, headers=headers, params=params, json=payload, timeout=timeout
            )
        except RequestException as e:  # Random errors may occur
            print(f"Request failed ({attempt + 1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:  # rate limited; honor Retry-After
            time.sleep(float(response.headers.get("Retry-After", 2)))
            continue
        if response.status_code >= 500:  # transient server error
            print(f"Server error {response.status_code} ({attempt + 1}/{retries})")
            time.sleep(2 * (attempt + 1))
            continue
        print(f"Request failed with status {response.status_code}")
        return None
    return None


def fetch_proteins(protein_ids: list[str], batch_size: int = 50) -> dict:
    """
    Fetch protein sequences for specific Ensembl protein IDs.

    Args:
        protein_ids (list[str]):
            Unversioned Ensembl protein IDs (ENSP*).
        batch_size (int):
            IDs per request (Ensembl REST allows at most 50).

    Returns:
        dict:
            Protein ID -> (version, sequence). IDs Ensembl does not return are absent.
    """

    proteins = {}
    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i : i + batch_size]
        data = _post_json(
            "https://rest.ensembl.org/sequence/id",
            {"ids": batch},
            params={"type": "protein"},
        )
        if data is None:
            print(f"Failed to fetch batch starting at {batch[0]}")
            continue
        for record in data:
            proteins[record["id"]] = (record.get("version"), record["seq"])
        print(f"Fetched {len(proteins)}/{len(protein_ids)} sequences")

    return proteins


def get_seqs(tf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe with sequences from ensembl.

    Args:
        tf_subset (pd.DataFrame):
            Subsetted TF dataframe.

    Returns:
        pd.DataFrame:
            TF dataframe with sequences.
    """

    tf_df["Sequence"] = tf_df["DBID"].apply(fetch_canonical_protein)
    tf_df = tf_df[tf_df["Sequence"].notna()]

    return tf_df


def main():
    """ """

    tf_df = pd.read_csv("data/sequences/selex/selex_candidates_tfs.csv")
    tf_df = get_seqs(tf_df)
    tf_df.to_csv("data/sequences/selex/selex_candidates_tfs.csv", index=False)


if __name__ == "__main__":
    main()
