# Blast

### Installation
You can install BLAST by following NCBI's guide https://www.ncbi.nlm.nih.gov/books/NBK569861/.

### Database
We create a database of all human proteins to BLASTP against. You can create this database with the following commands:

```
wget "ftp://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/human.*.protein.faa.gz"
gunzip human.*.protein.faa.gz
cat human.*.protein.faa > all_human_refseq_proteins.faa
makeblastdb -in all_human_refseq_proteins.faa -dbtype prot -out humandb/humandb
```

### Human TF sequences
The homeodomain and zinc-finger recognition-code comparisons read `data/sequences/animaltfdb/human_tfs.csv`. See the [AnimalTFDB sequence download instructions](../../sequences/animaltfdb/README.md) to regenerate this catalog from AnimalTFDB identifiers and Ensembl protein sequences.

### ZFIN orthology
The human counterpart of a zebrafish TF is taken from ZFIN's curated orthology.
Run `data/tools/blast/get_orthos.py` to download ZFIN's `human_orthos.txt` and `ensembl_1_to_1.txt` (cached under `zfin/`), reduce them to the TFs in our dataset, and write `zfin/ensdarg_human_orthos.csv`, an ensembl ID to human symbol mapping with a `jaspar_ids` column flagging which orthologs carry a JASPAR motif. Its similarity is recorded whether or not it has a JASPAR motif (`Hom_Sim` and `Hom_Gene` are always filled in, `Hom_JASPAR` only when a motif exists). Where a TF has more than one ZFIN ortholog, a motif-bearing one is preferred and ties break on similarity.

### Usage
We use BLASTP to measure how similar a zebrafish TF's DNA-binding domain is to that of its human ortholog. Zebrafish TFs with high DBD similarity to their human ortholog may share the motif of the human protein, so for these we can use a motif from JASPAR (which typically has information for human TFs) as the ground truth. You can run `tools/blast/run_blast.py` to search each zebrafish TF against the database of all human proteins created in the step above. Run `tools/blast/hom_results.py` to score the DBD region of the alignment against the ZFIN ortholog; `Hom_Sim` is the percent of DBD positions that are similar via BLOSUM62 substitution matrix scoring. The per-family similarity plot is in `notebooks/03_building-ppms/03_blast-hom-results.py`.

To note, we search the full length zebrafish protein sequence against the human database. We find that this gives higher similarities than searching just the DNA-binding domain, which may not be large enough to anchor the alignment to the correct ortholog.

For the families below, DNA-binding specificity is set by a small recognition code rather than by the domain as a whole, so `Hom_Sim` is the percent of matching residues in that code instead of a BLAST identity.

### C2H2 Zinc Fingers
Due to the number of zinc fingers in a typical zinc finger TF, and how their sequence specificity is determinant on the recognition code in each finger, we score these against the recognition code rather than by a blast search across the DNA-binding domain.

Only some of the fingers in an array actually engage DNA, and those are the fingers that set specificity, so the comparison is restricted to them. These DNA binding fingers are taken from the `DNA-Binding` flag in `tfdb4_rep_doms.csv`, written by `scripts/get_domains.py` from AF3 PAE protein-DNA contacts. Those fingers are then slid along the human ortholog's full array (no equivalent annotation exists for human TFs) and the best register is scored, with the spacing between non-adjacent binding fingers preserved.

Similarity is the fraction of matching residues at the four DNA-contacting positions (p-1, p2, p3, p6) of each binding finger — 12 of 16 matching residues over four fingers is 75%. A finger whose p-1 or p3 differs scores 0 of 4, since a substitution at either reassigns the base being read; a binding finger with no human partner at a register likewise scores 0 of 4, which is what makes registers hanging off the end of the human array lose. Run `data/tools/blast/zfc2h2_hom.py`.

### ZBTB
ZBTB TFs contain C2H2 zinc fingers that contact DNA and a BTB domain for protein-protein interactions. Like the zf-C2H2 family, we score them on recognition code similarity rather than full-sequence BLAST, using the same DNA-binding-finger metric described above. Run `data/tools/blast/zbtb_hom.py`.

### Homeodomains
Homeodomain containing TFs also have a recognition code system for DNA binding specificity, and we score similarity on these codes. The homeodomain is aligned to PF00046 and the residues at canonical positions 47, 50, 51, 52 and 54 form a 5-character code; `Hom_Sim` is the percent of those five that match the ortholog's code (5/5 is 100%, 4/5 is 80%). You can run `data/tools/blast/hd_hom.py`.
