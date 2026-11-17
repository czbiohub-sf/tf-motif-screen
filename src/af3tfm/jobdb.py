import glob
import json
import re
import sqlite3
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from tqdm import tqdm

from af3tfm.dataset import DS

# Predicted-contact thresholds for the contact-defined interface metrics
CONTACTED_BASE_MIN = 0.5
INTERFACE_CONTACT_MIN = 0.1


def _process_run_worker(args: tuple):
    """
    Worker for Database.process_runs — processes one AF3 output directory.

    Args:
        args: (direc_path, output_base, dom_csv)

    Returns:
        Tuple of (model_out, tf_name, dna_seq, motif_name, ptm, iptm,
        dom_plddt, dna_plddt, iface_pae, has_clash, fraction_disordered,
        ranking_score, prot_ptm, iface_pae_contact, iface_contact_prob,
        n_contacted_bases) or None if the job should be skipped.
    """

    try:
        return _process_run_worker_inner(args)
    except Exception:
        return None


def _process_run_worker_inner(args: tuple):
    direc_path, output_base, dom_csv = args
    direc = Path(direc_path)
    output_base = Path(output_base)

    # Parse TF and motif names
    pattern = (
        r"^(?P<protein>[a-z0-9]+(?:_\d+)?)_(?P<motif>.+?)_(?:top_)?(?P<seq_num>\d+)$"
    )
    match = re.match(pattern, direc.name)
    if not match:
        return None
    tf_name = match.group("protein")
    motif_name = match.group("motif")

    # Find valid output directory
    output_path = ""
    for d in (output_base / direc).iterdir():
        if d.is_dir() and any(d.iterdir()):
            for file in d.iterdir():
                if str(file).endswith("model.cif") and file.stat().st_size > 0:
                    output_path = output_base / direc / d.name
                    break
        if output_path:
            break
    if not output_path:
        return None
    output_path = Path(output_path)

    # Locate output files
    full_conf_out = ""
    for file in output_path.iterdir():
        if str(file).endswith("_model.cif"):
            model_out = str(output_path / file.name)
        if str(file).endswith("_summary_confidences.json"):
            summary_out = str(output_path / file.name)
        elif str(file).endswith("_confidences.json"):
            full_conf_out = str(output_path / file.name)

    # Confidence scores
    with open(summary_out) as f:
        summary_data = json.load(f)
    ptm, iptm = summary_data["ptm"], summary_data["iptm"]

    # Parse chain types and lengths from CIF
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", model_out)
    mmcif_dict = parser._mmcif_dict

    entity_ids = mmcif_dict["_entity_poly.entity_id"]
    entity_types = mmcif_dict["_entity_poly.type"]
    entity_type_map = {eid: etype for eid, etype in zip(entity_ids, entity_types)}

    chain_to_entity = {
        cid: eid
        for cid, eid in zip(
            mmcif_dict["_struct_asym.id"], mmcif_dict["_struct_asym.entity_id"]
        )
    }

    chain_list = list(structure[0].get_chains())
    chain_etypes = [
        entity_type_map.get(chain_to_entity.get(c.id, ""), "") for c in chain_list
    ]
    chain_lengths = [len(list(c.get_residues())) for c in chain_list]

    protein_chain_indices = [
        i for i, t in enumerate(chain_etypes) if "polypeptide" in t
    ]
    dna_chain_indices = [
        i for i, t in enumerate(chain_etypes) if "deoxyribonucleotide" in t
    ]

    if not protein_chain_indices:
        return None

    # Select protein chain with highest average chain_pair_iptm to DNA chains
    chain_pair_iptm = summary_data.get("chain_pair_iptm")
    if chain_pair_iptm and dna_chain_indices:
        if len(protein_chain_indices) > 1:
            best_protein_idx = max(
                protein_chain_indices,
                key=lambda pi: sum(chain_pair_iptm[pi][di] for di in dna_chain_indices)
                / len(dna_chain_indices),
            )
        else:
            best_protein_idx = protein_chain_indices[0]
        iptm = round(
            sum(chain_pair_iptm[best_protein_idx][di] for di in dna_chain_indices)
            / len(dna_chain_indices),
            3,
        )
    else:
        best_protein_idx = protein_chain_indices[0]

    # Interface PAE between the selected protein chain and the DNA chains.
    # chain_pair_pae_min[i][j] is the minimum expected position error at the
    # i-j interface (lower is better); average over DNA chains and take the
    # tighter of the two directions.
    chain_pair_pae_min = summary_data.get("chain_pair_pae_min")
    if chain_pair_pae_min and dna_chain_indices:
        pae_vals = [
            min(
                chain_pair_pae_min[best_protein_idx][di],
                chain_pair_pae_min[di][best_protein_idx],
            )
            for di in dna_chain_indices
        ]
        iface_pae = round(sum(pae_vals) / len(pae_vals), 3)
    else:
        iface_pae = 0.0

    # Tier-1 summary scalars (already loaded — no extra I/O).
    has_clash = float(summary_data.get("has_clash", 0.0))
    fraction_disordered = float(summary_data.get("fraction_disordered", 0.0))
    ranking_score = float(summary_data.get("ranking_score", 0.0))
    chain_ptm = summary_data.get("chain_ptm")
    prot_ptm = round(float(chain_ptm[best_protein_idx]), 3) if chain_ptm else 0.0

    # Map the selected chain back to its protein name for domain lookup.
    if "_homodimer" in output_path.name:
        lookup_name = tf_name
    else:
        protein_name_order = [tf_name, motif_name]
        chain_rank = protein_chain_indices.index(best_protein_idx)
        lookup_name = (
            protein_name_order[chain_rank]
            if chain_rank < len(protein_name_order)
            else tf_name
        )

    chain_offset = sum(chain_lengths[:best_protein_idx])
    chain_len = chain_lengths[best_protein_idx]

    # pLDDT scores
    plddt_scores = [float(x) for x in mmcif_dict["_ma_qa_metric_local.metric_value"]]

    dna_flat_indices, index = [], 0
    dna_res = ["DA", "DT", "DC", "DG"]
    for model in structure:
        for chain in model:
            for residue in chain:
                index += 1
                if residue.resname.strip() in dna_res:
                    dna_flat_indices.append(index)

    # pLDDT for DBD and DNA
    doms_df = pd.read_csv(dom_csv)
    dbd_rows = list(
        doms_df[(doms_df["DBID"] == lookup_name.upper()) & (doms_df["DNA-Binding"])][
            ["start", "end"]
        ].itertuples(index=False, name=None)
    )
    has_dbd = len(dbd_rows) > 0

    if has_dbd:
        # Average pLDDT across every position labeled DNA-binding (includes multiple DBDs)
        dbd_indices = set()
        for dbd_dom in dbd_rows:
            start = chain_offset + int(dbd_dom[0])
            end = chain_offset + int(dbd_dom[1])
            while start > chain_offset and plddt_scores[start - 1] > 70:
                start -= 1
            while end < chain_offset + chain_len and plddt_scores[end] > 70:
                end += 1
            while start < end and plddt_scores[start] < 70:
                start += 1
            while start < end and plddt_scores[end - 1] < 70:
                end -= 1
            dbd_indices.update(range(start, end))

        dbd_plddt = [plddt_scores[i] for i in sorted(dbd_indices)]
        dna_plddt_vals = [plddt_scores[i - 1] for i in dna_flat_indices]
        try:
            dom_plddt = round(sum(dbd_plddt) / len(dbd_plddt), 3)
            dna_plddt = round(sum(dna_plddt_vals) / len(dna_plddt_vals), 3)
        except ZeroDivisionError:
            return None
    else:
        dom_plddt, dna_plddt = 0.0, 0.0

    # Tier-2 interface metrics from the full (per-token) confidences file.
    iface_pae_contact, iface_contact_prob, n_contacted_bases = 0.0, 0.0, 0
    if full_conf_out:
        prot_tok = list(range(chain_offset, chain_offset + chain_len))
        dna_tok = [i - 1 for i in dna_flat_indices]

        if prot_tok and dna_tok:
            with open(full_conf_out) as f:
                conf_data = json.load(f)
            pae_mat = np.asarray(conf_data["pae"], dtype=float)
            contact = np.asarray(conf_data["contact_probs"], dtype=float)

            # contact_probs block: rows = DNA bases, cols = selected-chain residues.
            cblock = contact[np.ix_(dna_tok, prot_tok)]

            # Per-base best contact probability to any protein residue.
            per_base_max = cblock.max(axis=1)
            iface_contact_prob = round(float(per_base_max.mean()), 3)
            n_contacted_bases = int((per_base_max > CONTACTED_BASE_MIN).sum())

            # Interface PAE over the contact-defined interface: protein residues
            # whose best contact to any DNA base clears INTERFACE_CONTACT_MIN,
            # against all DNA bases (both matrix directions). Left at 0.0 when
            # no residue contacts the DNA.
            per_res_max = cblock.max(axis=0)
            iface_res = [
                prot_tok[j]
                for j in range(len(prot_tok))
                if per_res_max[j] > INTERFACE_CONTACT_MIN
            ]
            if iface_res:
                block = np.concatenate(
                    [
                        pae_mat[np.ix_(iface_res, dna_tok)].ravel(),
                        pae_mat[np.ix_(dna_tok, iface_res)].ravel(),
                    ]
                )
                iface_pae_contact = round(float(block.mean()), 3)

    # Sequences
    seq_entity_ids = mmcif_dict["_entity_poly_seq.entity_id"]
    seq_mon_ids = mmcif_dict["_entity_poly_seq.mon_id"]
    entity_residues = {}
    for eid, mon_id in zip(seq_entity_ids, seq_mon_ids):
        entity_residues.setdefault(eid, []).append(mon_id)

    dna_map = {"DA": "A", "DT": "T", "DC": "C", "DG": "G"}
    dna_seq = ""
    for eid, etype in entity_type_map.items():
        residues = entity_residues.get(eid, [])
        if "deoxyribonucleotide" in etype and not dna_seq:
            dna_seq = "".join(dna_map.get(res, "N") for res in residues)

    # Remove seed directories
    for seed in glob.glob(f"{output_path}/seed*"):
        for file in glob.glob(f"{seed}/*"):
            Path(file).unlink()
        Path(seed).rmdir()

    return (
        model_out,
        tf_name,
        dna_seq,
        motif_name,
        ptm,
        iptm,
        dom_plddt,
        dna_plddt,
        iface_pae,
        has_clash,
        fraction_disordered,
        ranking_score,
        prot_ptm,
        iface_pae_contact,
        iface_contact_prob,
        n_contacted_bases,
    )


class Database:
    """
    SQLite database used to store job information and keep state between runs.

    Attributes:
        db_path (str): Path to the SQLite database file.
        conn (sqlite3.Connection): SQLite connection object.
        cursor (sqlite3.Cursor): SQLite cursor object.
    """

    def __init__(
        self,
        job_path: str,
        output_path: str = None,
        dataset=str,
    ):
        """
        Initializes Database object and creates tables if they do not exist.

        Args:
            job_path (str): The path to the job directory.
            output_path (str): The path to the output directory containing AF3 predictions.
            dataset (str): The dataset being worked with.
        """

        self.path = Path(job_path)
        self.dataset = DS[dataset]

        self.db_path = f"{job_path}/jobdb.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if output_path is not None:
            self.output_path = Path(output_path)

        run_table = """CREATE TABLE IF NOT EXISTS runs (
                structure_path text PRIMARY KEY,
                protein text NOT NULL,
                dna text NOT NULL,
                motif text NOT NULL,
                ptm real NOT NULL,
                ipTM real NOT NULL,
                dom_pLDDT real NOT NULL,
                dna_pLDDT real NOT NULL,
                iface_PAE real NOT NULL,
                has_clash real NOT NULL,
                fraction_disordered real NOT NULL,
                ranking_score real NOT NULL,
                prot_pTM real NOT NULL,
                iface_PAE_contact real NOT NULL,
                iface_contact_prob real NOT NULL,
                n_contacted_bases integer NOT NULL
                ); """

        self.cursor.execute(run_table)
        self.conn.commit()

    def process_runs(self, max_workers: int = None, batch_size: int = 1000):
        """
        Read directories in self.output_path and updates the database with job information.
        Commits every batch_size rows so progress is saved if the job is interrupted.

        Args:
            max_workers: Number of worker processes (None = CPU count).
            batch_size: Number of rows to insert between commits.
        """

        args_list = [
            (str(direc), str(self.output_path), str(self.dataset.dom_csv))
            for direc in self.output_path.iterdir()
        ]

        insert = """
            INSERT INTO runs (
                structure_path, protein, dna, motif, ptm, ipTM, dom_pLDDT, dna_pLDDT,
                iface_PAE, has_clash, fraction_disordered, ranking_score, prot_pTM,
                iface_PAE_contact, iface_contact_prob, n_contacted_bases
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(structure_path) DO UPDATE SET
                dom_pLDDT = excluded.dom_pLDDT,
                dna_pLDDT = excluded.dna_pLDDT,
                iface_PAE = excluded.iface_PAE,
                has_clash = excluded.has_clash,
                fraction_disordered = excluded.fraction_disordered,
                ranking_score = excluded.ranking_score,
                prot_pTM = excluded.prot_pTM,
                iface_PAE_contact = excluded.iface_PAE_contact,
                iface_contact_prob = excluded.iface_contact_prob,
                n_contacted_bases = excluded.n_contacted_bases;
            """

        n = len(args_list)
        self.conn.close()  # Close connection before forking

        try:
            results = []
            with Pool(processes=max_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_process_run_worker, args_list, chunksize=10),
                    total=n,
                    desc="Processing jobs",
                ):
                    if result:
                        results.append(result)
        finally:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

        pending = 0
        for result in results:
            self.cursor.execute(insert, result)
            pending += 1
            if pending >= batch_size:
                self.conn.commit()
                pending = 0
        self.conn.commit()
