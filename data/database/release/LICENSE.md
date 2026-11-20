# $title

## Data rights and notices

This dataset follows the licensing policy documented by
[ZebraMotif](https://github.com/czbiohub-sf/zebramotif-explorer#license-and-policies) and
[$pipeline_name]($pipeline_url#alphafold3-outputs).
Their MIT licenses apply to software. AlphaFold3 predictions and data derived
from them are provided under and subject to the
[AlphaFold3 Output Terms of Use](notices/AF3_OUTPUT_TERMS_OF_USE.md), including
their use restrictions, notice requirements, and citation requirement.

## Scope by file

| Files or content | Applicable rights and notices |
|---|---|
| `structures/*.cif.gz` | AlphaFold3 Output Terms of Use. Original notices remain inside every structure. |
| Predicted matrices and AF3-derived fields in `pwm_modes.parquet`, both `.meme` exports, and collection metadata TSVs | AlphaFold3 Output Terms of Use, following the repositories' policy for derived specificity predictions. |
| `confidence_metrics.parquet` and `interface_residues.parquet` | AlphaFold3 Output Terms of Use. |
| `dbd_domains.parquet` | Domain identities, descriptions, and coordinates derive from InterPro/Pfam/PROSITE annotations. The `DNA-Binding` placement flags additionally derive from AF3 predictions and follow the AlphaFold3 Output Terms of Use. |
| Catalog IDs, family labels, gene symbols, and ZFIN/JASPAR references in the Parquet tables | Retain attribution to the source resources listed below and their applicable terms. This release assigns no additional blanket license to third-party reference annotations. |
| ZebraMotif and $pipeline_name software | MIT, with the original copyright holders and years preserved in `notices/ZebraMotif-MIT.txt` and `$pipeline_notice`. The source repositories distribute the software. |

The AlphaFold3 Output Terms include non-commercial-use restrictions and define
Output to include computational predictions and adaptations substantially
derived from it. A standard MIT, CC0, CC BY, or CC BY-NC label alone does not
describe these custom terms. Retain the accompanying
[Legally Binding Terms of Use](Legally%20Binding%20Terms%20of%20Use.txt) notice,
the full terms, and the AlphaFold3 paper citation when redistributing the
predictions and derivatives.

Recipients who have agreed to the AlphaFold3 Model Parameters Terms of Use
should consult those terms as specified by the Output Terms. Copies of the
[Model Parameters Terms](notices/AF3_WEIGHTS_TERMS_OF_USE.md) and their
[Prohibited Use Policy](notices/AF3_WEIGHTS_PROHIBITED_USE_POLICY.md) accompany
this release for reference. The applicable version is governed by the upstream
terms. Model parameters are obtained directly from Google DeepMind.

## Processing and modifications

The release gzip-compresses and names selected AF3 structures by motif ID.
It summarizes AF3 confidence metrics and residue contacts, derives specificity
matrices with NA-MPNN, clusters those matrices into consensus modes, compares
them with JASPAR references, and serializes them as Parquet and MEME files.
It joins reference identifiers and domain annotations to those predictions.
`README.md`, `SCHEMA.md`, and `PROVENANCE.json` describe the processing,
selection settings, source versions, and coverage.

## Source resources

Reference annotations retain the rights of their original providers. Consult
the cited resource for the applicable terms and cite the resources used in
your analysis.

| Resource | Contributions and attribution |
|---|---|
| [AnimalTFDB 4.0](https://guolab.wchscu.cn/AnimalTFDB4/) | Zebrafish TF catalog and family labels; Shen et al., *Nucleic Acids Research* (2023). |
| [JASPAR 2024](https://jaspar.elixir.no/) | Reference motif identifiers and comparison inputs; Rauluseviciute et al., *Nucleic Acids Research* (2024). |
| [InterPro](https://www.ebi.ac.uk/interpro/) and [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/) | Domain identities, descriptions, and coordinates from the pipeline's InterProScan annotations; CC0 1.0 for the downloadable InterPro and Pfam data. |
| [PROSITE](https://prosite.expasy.org/) | The retained 11 domain rows with `PS` accessions in `dbd_domains.parquet`, from PROSITE Profiles `2023_05` via InterProScan. Copyright SIB Swiss Institute of Bioinformatics; [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), subject to the [PROSITE license conditions](https://prosite.expasy.org/prosite_license.html). |
| [ZFIN](https://zfin.org/) | Gene identifiers and gene-page links from the frozen Ensembl-to-ZFIN mapping. |
| [Ensembl](https://www.ensembl.org/Danio_rerio/Info/Index) | Gene identifiers used to join the catalog, predictions, and annotations. |
| [NA-MPNN](https://github.com/baker-laboratory/NA-MPNN) | Specificity prediction method applied to AF3 structures; software/checkpoints are obtained from its source project. |

## PROSITE attribution and license

The 11 retained PROSITE annotation rows in `dbd_domains.parquet` are attributed
to the SIB Swiss Institute of Bioinformatics, the copyright holder of PROSITE.
The release metadata records PROSITE Profiles `2023_05`, used through
InterProScan `5.73-104.0`. PROSITE annotations are subject to the
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
license](https://creativecommons.org/licenses/by-nc-nd/4.0/) and the provider's
[license conditions](https://prosite.expasy.org/prosite_license.html), including
non-commercial and no-derivatives restrictions. Preserve this attribution and
license information when redistributing these annotations.

The `DNA-Binding` placement flags joined to these domain rows are derived from
AF3 predictions and additionally follow the AlphaFold3 Output Terms of Use.
Neither the software's MIT license nor the AF3 terms replace the PROSITE terms
for the reference annotations.

## AlphaFold3 citation

Abramson, J. et al. Accurate structure prediction of biomolecular interactions
with AlphaFold 3. *Nature* **630**, 493–500 (2024).
[https://doi.org/10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)
