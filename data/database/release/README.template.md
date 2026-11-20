# $title

ZebraMotif provides predicted DNA-binding probability matrices and selected
protein-DNA structures for zebrafish (*Danio rerio*) transcription factors.
This is dataset version **$version**: **$release_label**.

AlphaFold3 predictions and derivatives are provided under the
[AlphaFold3 Output Terms of Use](notices/AF3_OUTPUT_TERMS_OF_USE.md).
Read [LICENSE.md](LICENSE.md) and
[Legally Binding Terms of Use.txt](Legally%20Binding%20Terms%20of%20Use.txt)
before reuse. The repositories' MIT software licenses apply to their code.

## Contents

| File or directory | Contents |
|---|---|
| `pwm_modes.parquet` | $modes binding modes for $primary_modes modeled proteins: $primary_modes primary and $alternate_modes alternate modes, including matrices, support, reference comparisons, and structure paths. |
| `tf_to_rep.parquet` | $query_entries query entries: $full_length_entries full-length gene entries and $domain_entries domain entries; $resolved_queries entries resolve to a representative motif. |
| `structures/` | $structures selected AlphaFold3 structures, one gzip-compressed mmCIF file per primary mode. |
| `interface_residues.parquet` | $contact_rows DNA-contact residue annotations for $contact_proteins proteins. |
| `confidence_metrics.parquet` | $confidence_rows confidence summary rows for $confidence_proteins proteins, across all predictions and the top 47. |
| `dbd_domains.parquet` | $domain_rows domain annotations for $annotated_proteins proteins, including DNA-placement flags. |
| `zebramotif_primary_modes.meme` | $primary_modes primary probability matrices in MEME format. |
| `zebramotif_all_modes.meme` | All $modes probability matrices in MEME format. |
| `zebramotif_{primary,all}_modes.metadata.tsv` | Query-to-mode mappings, sequence types, raw metrics, and the symbol labels used in MEME files. |
| `SCHEMA.md` | Column definitions, identifiers, units, joins, missing values, and score interpretation. |
| `release_metadata.json` | Dataset and source versions, counts, methods, citation, and data-use terms. |
| `PROVENANCE.json` | File sizes and SHA-256 hashes, observed repository revisions and source files, pipeline settings, and coverage limitations. |
| `CITATION.cff` | Machine-readable dataset citation. |
| `ZENODO_METADATA.json` | Dataset description, creators, related resources, and scoped license metadata for the deposit. |
| `notices/` | AlphaFold3 terms and preserved software license notices. |
| `SHA256SUMS` | Checksums of every distributed file except the checksum file itself. |

The $primary_modes modeled representatives comprise $full_length_representatives
full-length proteins and $domain_representatives separately modeled domains.
Domain entries and annotation coverage are listed in `PROVENANCE.json`.

This release contains selected primary-mode structures and summary data.
Full screening ensembles and upstream reference databases are separate resources.

## Methods and sources

The [$pipeline_name pipeline]($pipeline_url)
folds proteins with candidate DNA using AlphaFold3, predicts binding
specificities with NA-MPNN, and clusters them into consensus modes. This export
uses **47** selected specificities, a **0.73** UPGMA merge cutoff, up to three
alternates, and a **0.72** similarity cutoff for `primary_refined` alternates.
Each primary structure is selected from its mode using the ipTM × DBD-pLDDT
rank product, where DBD-pLDDT is the mean pLDDT over the DNA-binding domain.

AnimalTFDB 4.0 supplies catalog identifiers and family labels. JASPAR 2024 CORE
vertebrate motifs supply reference comparisons. InterProScan annotations from
Pfam and PROSITE Profiles supply domain descriptions and coordinates;
DNA-placement flags use AF3 evidence. ZFIN annotations identify query genes.
Source identities and coverage are recorded in `PROVENANCE.json`; resource links
and attributions appear in [LICENSE.md](LICENSE.md).

## Citation

$citation

Please also cite Abramson, J. et al. **Accurate structure prediction of
biomolecular interactions with AlphaFold 3.** *Nature* **630**, 493–500 (2024).
[doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)

Cite the source resources relevant to your analysis, including AnimalTFDB
([Shen et al., 2023](https://guolab.wchscu.cn/AnimalTFDB4/)) and JASPAR
([Rauluseviciute et al., 2024](https://jaspar.elixir.no/)).

## Integrity

After extracting the dataset with its relative paths preserved, run:

```bash
sha256sum -c SHA256SUMS
```

These checksums describe the extracted release files. The exporter also supplies
archive checksums beside the full dataset and primary/all motif ZIPs.
