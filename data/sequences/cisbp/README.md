# Cis-BP

### Transcription Factors
Zebrafish transcription factor motifs are pulled from CisBP (build `3_10`) https://cisbp.ccbr.utoronto.ca/.
1. Run `data/sequences/cisbp/get_cisbp_danio.py` to download the full Danio_rerio TF table and all of its motifs (direct and inferred). This writes the TF table to `data/sequences/cisbp/TF_Information.txt`, each PWM to `data/sequences/cisbp/pwms/<Motif_ID>.txt`, and lists the retrieved motifs (`dbid,tf_name,motif_id`) in `top_cisbp_motifs.txt`.
2. Run `data/sequences/cisbp/get_cisbp_seqs.py` to build the curated TF set. It slices the hand-picked TFs (`CURATED_TFS` in that script) out of `TF_Information.txt`, fetches each canonical protein sequence from Ensembl, annotates the AnimalTFDB4 family, and writes `data/sequences/cisbp/cisbp_tfs.csv` (carrying the build-`3_10` `Motif_ID` for each TF). The curated set is the direct-evidence Danio TFs plus a few added for DNA-binding-domain diversity (high homology to human proteins with JASPAR motifs; see data/tools/blast), excluding a handful that are unusable (empty motif, no Ensembl ID, or too large to fold).
3. Run `data/sequences/cisbp/get_motifs.py` to assemble the per-TF experiment PWMs into `data/sequences/cisbp/motifs/`, kept separate from the raw CisBP `pwms/` dump. The motif source depends on the CisBP evidence level (`TF_Status` for the exact gene+`Motif_ID` pair in `TF_Information.txt`): direct-evidence TFs (`D`) use their own CisBP PWM copied verbatim from `pwms/<Motif_ID>.txt`; inferred TFs (`I`) instead use the JASPAR motif of their homologous human TF, whose matrix ID is read from the `Hom_JASPAR` column of `data/sequences/animaltfdb/tfdb4_rep_tfs.csv`  and whose PWM is pulled from `data/tools/homer/jaspar.meme.motif` and converted to the CisBP tab format. This writes one `motifs/<motif_id>.txt` per TF and a `dbid,tf_name,motif_id` manifest to `motifs.txt`.

### Motifs
In these runs we are looking to compare AlphaFold3 predictions between sequences generated from a TF's motif and non-motif sequences. Each TF's motif PWM is the `motifs/<motif_id>.txt` file from step 3 (CisBP for direct-evidence TFs, JASPAR for inferred ones), and the TF -> motif assignment is the `motifs.txt` manifest.

### Non-motifs
The non-motif (negative) sequences are GC-matched genomic k-mers drawn from ATAC peaks, screened so they cannot look like a real binding site (below the PWM's p<1e-3 log-odds threshold and free of the motif's high-IC core seed on either strand). The screening PWM is each TF's `motifs/<motif_id>.txt` from `motifs.txt`, so inferred TFs are screened against the same JASPAR motif they are tested against. Run `python data/sequences/cisbp/get_nonmotif_seqs.py` to generate them; per-TF fastas are written to `data/sequences/cisbp/nonmotif_negatives/<TF>_negatives.fa` 

### Running jobs
To predict the structures of the CisBP TFs with DNA sequences sampled from their motifs (positives) and the genomic non-motif negatives, run the following commands:

1. `python scripts/make_json.py --job_type cpu --job_dir data/jobs/motif-nonmotif/input_data --dataset cisbp`
2. `python scripts/run_af3.py --input_path data/jobs/motif-nonmotif/input_data --dataset cisbp --job_type cpu --partition cpu --output_path data/jobs/motif-nonmotif/output_data`
3. `python scripts/make_json.py --job_type gpu --job_dir data/jobs/motif-nonmotif-genomic/input_fold --dataset cisbp --motif_num 30 --top_motifs data/sequences/cisbp/motifs.txt --pwm_path data/sequences/cisbp/motifs --data_path data/jobs/motif-nonmotif/output_data --neg_dir data/sequences/cisbp/nonmotif_negatives`
4. `python scripts/run_af3.py --input_path data/jobs/motif-nonmotif-genomic/input_fold --dataset cisbp --job_type gpu --partition preempted --output_path data/jobs/motif-nonmotif-genomic/output_fold`

The `--neg_dir` flag adds one GPU job per non-motif sequence (named `{dbid}_{fasta_header}`) alongside the positive jobs sampled from the PWMs. The protein MSA/template data from the CPU step (`data/jobs/motif-nonmotif/output_data`) is reused for both.

### Processing structures and confidence scores for analysis
Following structure prediction, we perform a few processing steps. First we determine the DNA binding domain from the lowest PAE regions between the predicted protein-DNA complexes that match with interproscan predicted domain regions. Then, we extract the confidence metrics from the AF3 structure predictions and store them in a sqlite database. From the most confident predictions per TF we then determine which residues are closest in contact to the DNA. Finally, we add another table to the sqlite database which stores PAE values for these contacts for each prediction.

1. `python scripts/get_domains.py --dataset cisbp --fold-dirs data/jobs/motif-nonmotif-genomic/output_fold`
2. `python scripts/proc_runs.py --dataset cisbp --run motif-nonmotif-genomic --direc fold --table runs`
3. `python data/get_contacts.py --dataset cisbp --run motif-nonmotif-genomic --direc fold`
4. `python scripts/proc_runs.py --dataset cisbp --run motif-nonmotif-genomic --direc fold --table contact_pae`
