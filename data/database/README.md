# ZebraMotif database

We use this directory to build and package the predicted zebrafish TF motif database
for ZebraMotif and Zenodo. The export includes primary and alternate binding modes,
TF-to-representative mappings, selected AF3 structures, confidence summaries,
DNA-contact residues, and domain annotations.

### Usage

Run from the repository root:

```bash
python data/database/export_data.py                  # build and package
python data/database/export_data.py --from-existing  # package existing data
python data/database/export_data.py --check          # verify the export
```

Set the dataset version, creators, DOI, and source information in
`release/metadata.json`. Templates and notices are also kept in `release/`;
`export_tools/` contains the supporting code and `tests/` contains its tests.

Packaging checks that every gene ID in the original AnimalTFDB4 catalog
(`data/sequences/animaltfdb/tfdb4_tfs.csv`) resolves to an exported motif.
Missing or unresolved catalog entries stop the export.

### Maintaining the scientific tables

These options refresh existing tables without rebuilding motif predictions:

| Option | Refresh |
|---|---|
| `--bands-only` | Descriptive score bands. |
| `--lookup-only` | Representative self-resolution and MEME labels. |
| `--metadata-only` | Lookup, reference annotations, and domain metadata. |
| `--domains-only` | Domain annotations and DNA-placement flags. |
| `--confidence-only` | Per-protein confidence summaries. |

Each command also packages the complete release. Metadata refreshes preserve
the existing motif matrices and other query assignments; they do not rerun
sequence clustering. Website display and deployment settings belong in the
website repository.

### Output

- `export/database/`: working Parquet tables, MEME files, and compressed structures.
- `export/<version>/raw/`: the complete dataset with metadata, documentation,
  citation, provenance, and notices.
- `export/<version>/`: the full dataset ZIP, primary/all motif ZIPs, and checksums.

### Zenodo handoff

Before preparing the final upload, update `release/metadata.json` and
`release/LICENSE.md` to point to the created public-source snapshot repository,
including its issue tracker and README sections. Then regenerate the release
and verify its documents, archives, and checksums:

```bash
python data/database/export_data.py --from-existing
python data/database/export_data.py --check
```

Upload `zebramotif-data-<version>.zip` and its `.zip.sha256` file to Zenodo.
Set the record's license to **Other** and direct readers to the deposited
`LICENSE.md` and the
[AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).
The generated `raw/ZENODO_METADATA.json` provides custom rights titled `Other`
and describes the component-specific terms; it does not change the live record.
Do not select MIT or a Creative Commons license for the dataset as a whole.

Before publication, verify the live record's license and links, download its
uploaded archive and checksum, and compare them with the verified local release.
Check that the archive contains the corrected `LICENSE.md`, `Legally Binding
Terms of Use.txt`, and preserved notices. Archive generation alone does not
verify or publish the Zenodo record.
