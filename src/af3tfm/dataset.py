from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Dataset:
    """
    Stores information about sequences and their .csv files.
    """

    name: str
    seq_csv: str = None
    dom_csv: str = None


cisbp = Dataset(
    name="cisbp",
    seq_csv=f"{DATA_DIR}/sequences/cisbp/cisbp_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/cisbp/cisbp_doms.csv",
)

selex_candidates = Dataset(
    name="selex_candidates",
    seq_csv=f"{DATA_DIR}/sequences/selex/selex_candidates_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/selex/selex_candidates_doms.csv",
)

selex = Dataset(
    name="selex",
    seq_csv=f"{DATA_DIR}/sequences/selex/selex_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/selex/selex_doms.csv",
)

tfdb4 = Dataset(
    name="tfdb4",
    seq_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv",
)

tfdb4_rep = Dataset(
    name="tfdb4_rep",
    seq_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_doms.csv",
)

tfdb4_val = Dataset(
    name="tfdb4_val",
    seq_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_val_tfs.csv",
    dom_csv=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_val_doms.csv",
)

DS: dict[str, Dataset] = {
    "cisbp": cisbp,
    "selex_candidates": selex_candidates,
    "selex": selex,
    "tfdb4": tfdb4,
    "tfdb4_rep": tfdb4_rep,
    "tfdb4_val": tfdb4_val,
}
