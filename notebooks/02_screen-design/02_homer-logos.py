# %%
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import parse_pwm_file
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% Define motifs to plot: list of (file, motif_name, label)
motifs_to_plot = [
    (
        f"{DATA_DIR}/tools/homer/jaspar_clustered/nonRedundant.motifs",
        "5-ACGCGAAC,BestGuess:MA1650.2(0.706),BestGuess:SWI4/MA0401.1/Jaspar(0.855)",
        "HOMER de novo (ACGCGAAC)",
    ),
    (
        f"{DATA_DIR}/tools/homer/jaspar.homer.motif",
        "MA1650.1",
        "MA1650.1 (ZBTB14)",
    ),
]

cols = list("ACGT")


def pwm_to_information(matrix: np.ndarray) -> pd.DataFrame:
    """Convert a probability matrix to information content (bits)."""
    bg = np.array([0.25, 0.25, 0.25, 0.25])
    p = np.clip(matrix, 1e-9, 1)
    ic_per_pos = np.sum(p * np.log2(p / bg), axis=1, keepdims=True)
    ic_matrix = p * ic_per_pos
    return pd.DataFrame(ic_matrix, columns=cols)


def load_motif(file: str, name: str) -> np.ndarray:
    pwms = parse_pwm_file(Path(file))
    if name not in pwms:
        raise KeyError(
            f"Motif '{name}' not found in {file}.\nAvailable: {list(pwms.keys())[:10]}"
        )
    return pwms[name][1]  # [consensus, matrix] -> matrix


# %% Plot
fig, axes = plt.subplots(len(motifs_to_plot), 1, figsize=(8, 2 * len(motifs_to_plot)))
if len(motifs_to_plot) == 1:
    axes = [axes]

for ax, (file, name, label) in zip(axes, motifs_to_plot):
    matrix = load_motif(file, name)
    ic_df = pwm_to_information(matrix)
    logomaker.Logo(ic_df, ax=ax, color_scheme="classic")
    ax.set_title(label)
    ax.set_ylabel("Bits")
    ax.set_xlabel("Position")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/motif_logos.svg", dpi=300)
plt.show()

# %%
