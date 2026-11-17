"""Download the CisBP bulk TF table and extract Danio rerio TFs that have a
binding motif (direct or inferred), one row per TF gene. CisBP has no REST API,
this pulls the whole-database bulk dump instead and filters to Danio_rerio.

  TF_Information.txt   one row per Danio TF gene that has a motif (best motif)
  pwms/<Motif_ID>.txt  position weight matrix for each retained motif
  top_cisbp_motifs.txt one "dbid,tf_name,motif_id" line per retrieved motif

Ben Iovino 8/6/2026 Biohub
"""

import urllib.request
import zipfile

import pandas as pd

from af3tfm.dataset import DATA_DIR

BUILD = "3_10"
BASE = f"https://cisbp.ccbr.utoronto.ca/data/{BUILD}/DataFiles/Bulk_downloads/EntireDataset"
BULK_ZIP = "TF_Information.txt.zip"
PWM_ZIP = "PWMs.zip"  # per-motif PWM text files under pwms/<Motif_ID>.txt

HERE = DATA_DIR / "sequences/cisbp"


def main() -> None:
    zip_path = HERE / BULK_ZIP
    txt_path = HERE / BULK_ZIP.removesuffix(".zip")

    print(f"Downloading {BULK_ZIP} ...")
    urllib.request.urlretrieve(f"{BASE}/{BULK_ZIP}", zip_path)

    # Read the bulk table straight from the archive
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(txt_path.name) as fh:
            df = pd.read_csv(fh, sep="\t", dtype=str, low_memory=False)
    dr = df[df["TF_Species"] == "Danio_rerio"].copy()

    # A TF has a motif iff Motif_ID is present (motif-less rows use ".")
    has_motif = dr[dr["Motif_ID"].notna() & (dr["Motif_ID"] != ".")]
    has_motif = has_motif.sort_values(
        "TF_Status"
    )  # Prefer direct (D) over inferred (I)
    genes = has_motif.drop_duplicates("TF_ID")
    genes.to_csv(HERE / "TF_Information.txt", sep="\t", index=False)

    # Pull the PWM for each retained motif out of the bulk PWM archive
    pwm_zip_path = HERE / PWM_ZIP
    pwm_dir = HERE / "pwms"
    pwm_dir.mkdir(exist_ok=True)

    print(f"Downloading {PWM_ZIP} ...")
    urllib.request.urlretrieve(f"{BASE}/{PWM_ZIP}", pwm_zip_path)
    saved: set[str] = set()
    with zipfile.ZipFile(pwm_zip_path) as zf:
        for motif_id in genes["Motif_ID"].unique():
            member = f"pwms/{motif_id}.txt"
            try:
                data = zf.read(member)
            except KeyError:
                print(f"  warning: no PWM file for {motif_id}")
                continue
            # CisBP ships some motifs as a header-only file, skip them
            rows = [ln for ln in data.decode().splitlines() if ln.strip()]
            if len(rows) <= 1:
                print(f"  warning: empty PWM for {motif_id}, skipping")
                continue
            (pwm_dir / f"{motif_id}.txt").write_bytes(data)
            saved.add(motif_id)

    # List every retrieved motif (dbid,tf_name,motif_id)
    have_pwm = genes[genes["Motif_ID"].isin(saved)]
    with open(HERE / "top_cisbp_motifs.txt", "w") as fh:
        for dbid, tf_name, motif_id in zip(
            have_pwm["DBID"], have_pwm["TF_Name"], have_pwm["Motif_ID"]
        ):
            fh.write(f"{dbid.lower()},{tf_name},{motif_id}\n")

    zip_path.unlink()
    pwm_zip_path.unlink()


if __name__ == "__main__":
    main()
