import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from export_tools.database_export import DOMAIN_COLUMNS
from export_tools.release_export import TABLES, export_release, inventory


class ReleaseExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "pipeline"
        self.root.mkdir()
        project = Path(__file__).resolve().parents[1]
        self.export_root = self.root / "data/database"
        shutil.copytree(project / "release", self.export_root / "release")
        (self.export_root / "export_data.py").write_text("# Fixture export code\n")
        metadata_path = self.export_root / "release/metadata.json"
        config = json.loads(metadata_path.read_text())
        config["version"] = "0.1"
        metadata_path.write_text(json.dumps(config))
        shutil.copy2(project.parents[1] / "LICENSE", self.root / "LICENSE")
        (self.root / "README.md").write_text("Fixture pipeline\n")
        self.database = self.export_root / "export/database"
        self.database.mkdir(parents=True)
        self.output = self.export_root / "export/0.1"
        self.gene = "ENSDARG00000010001"
        self.domain = self.gene + "_1"
        self.modes = pd.DataFrame(
            [
                self.mode(self.gene, "alpha", "ZM00000010001.1", "primary"),
                self.mode(self.gene, "alpha", "ZM00000010001.2", "alternate"),
                self.mode(self.domain, "alpha_domain1", "ZM000000100011.1", "primary"),
            ]
        )
        self.modes.to_parquet(self.database / "pwm_modes.parquet", index=False)
        self.queries = pd.DataFrame(
            {
                "DBID": [self.gene, self.domain, "MEMBER", "UNRESOLVED"],
                "TF_Name": ["alpha", "alpha_domain1", "member", "unresolved"],
                "Family": ["family"] * 4,
                "sequence_type": [
                    "full_length",
                    "domain",
                    "full_length",
                    "full_length",
                ],
                "Rep_DBID": [self.domain, self.domain, self.gene, None],
                "is_representative": [False, True, False, False],
                "resolution": ["cluster", "self", "cluster", "unresolved"],
                "zfin_id": ["ZDB-GENE-1", None, "ZDB-GENE-2", None],
                "in_original_catalog": [True, False, True, True],
            }
        )
        self.queries.to_parquet(self.database / "tf_to_rep.parquet", index=False)
        pd.DataFrame({"DBID": [self.gene], "residue_num": [1]}).to_parquet(
            self.database / "interface_residues.parquet", index=False
        )
        pd.DataFrame(
            {"DBID": [self.gene], "subset": ["all"], "metric": ["ipTM"], "mean": [0.8]}
        ).to_parquet(self.database / "confidence_metrics.parquet", index=False)
        pd.DataFrame(
            [
                [self.gene, "alpha", "IPR1", "Domain", 1, 2, True],
                [self.domain, "alpha_domain1", "IPR1", "Domain", 1, 2, False],
            ],
            columns=DOMAIN_COLUMNS,
        ).to_parquet(self.database / "dbd_domains.parquet", index=False)
        (self.database / "structures").mkdir()
        for path in self.modes.structure_path.dropna():
            (self.database / path).write_bytes(
                gzip.compress(b"data_test\n# Original AF3 notice\n", mtime=0)
            )
        (self.database / "annotation_audit").mkdir()
        (self.database / "annotation_audit/private_research.txt").write_text(
            "Not a release deliverable\n"
        )

    def mode(self, dbid, symbol, motif, tier):
        return {
            "DBID": dbid,
            "TF_Name": symbol,
            "Family": "family",
            "motif_id": motif,
            "sequence_type": "domain" if dbid.endswith("_1") else "full_length",
            "tier": tier,
            "mode_rank": 1,
            "n_members": 20 if tier == "primary" else 5,
            "num_specs": 47,
            "cor_threshold": 0.73,
            "pwm_length": 2,
            "pwm": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            "structure_path": f"structures/{motif}.cif.gz"
            if tier == "primary"
            else None,
            "corr_to_primary": 0.5 if tier == "alternate" else None,
            "offset_to_primary": 2 if tier == "alternate" else None,
            "rc_to_primary": False if tier == "alternate" else None,
            "hom_jaspar": None,
            "hom_sim": None,
            "homer_sim": None,
            "z_score": None,
            "p_value": None,
            "recovery": None,
        }

    def export(self, **kwargs):
        return export_release(
            self.root, self.database, self.output, schema="# Fixture schema\n", **kwargs
        )

    def test_incomplete_catalog_coverage_does_not_replace_an_existing_release(self):
        self.export()
        before = inventory(self.output)
        catalog = self.root / "data/sequences/animaltfdb/tfdb4_tfs.csv"
        catalog.parent.mkdir(parents=True)
        for missing_id in ["ENSDARG00000052618", "UNRESOLVED"]:
            with self.subTest(dbid=missing_id):
                pd.DataFrame({"DBID": [self.gene, missing_id]}).to_csv(
                    catalog, index=False
                )
                with self.assertRaisesRegex(ValueError, "Incomplete AnimalTFDB4"):
                    self.export()
                self.assertEqual(inventory(self.output), before)

    def verify_archive_checksums(self, archive):
        manifest = dict(
            line.split("  ", 1)[::-1]
            for line in archive.read("SHA256SUMS").decode().splitlines()
        )
        self.assertEqual(set(manifest), set(archive.namelist()) - {"SHA256SUMS"})
        for name, digest in manifest.items():
            self.assertEqual(
                hashlib.sha256(archive.read(name)).hexdigest(), digest, name
            )

    def test_all_raw_and_archive_deliverables_are_generated_without_website_sources(
        self,
    ):
        before = inventory(self.database)
        self.export()
        raw = self.output / "raw"
        self.assertEqual(inventory(self.database), before)
        for name in [
            *TABLES,
            "SCHEMA.md",
            "release_metadata.json",
            "README.md",
            "LICENSE.md",
            "CITATION.cff",
            "ZENODO_METADATA.json",
            "PROVENANCE.json",
        ]:
            self.assertTrue((raw / name).is_file(), name)
        self.assertFalse((raw / "annotation_audit").exists())
        with zipfile.ZipFile(self.output / "zebramotif-data-0.1.zip") as archive:
            self.assertEqual(set(archive.namelist()), set(inventory(raw)))
            self.verify_archive_checksums(archive)
            citation = yaml.safe_load(archive.read("CITATION.cff"))
            self.assertEqual(citation["type"], "dataset")
            (paper,) = citation["references"]
            self.assertEqual(paper["type"], "article")
            self.assertEqual(paper["doi"], "10.64898/2026.11.18.767614")
            metadata = json.loads(archive.read("release_metadata.json"))
            self.assertEqual(metadata["citation"]["project_paper"]["doi"], paper["doi"])
            pipeline_notice = "notices/tf-motif-screen-MIT.txt"
            self.assertEqual(
                archive.read(pipeline_notice), (self.root / "LICENSE").read_bytes()
            )
            self.assertIn(f"`{pipeline_notice}`", archive.read("LICENSE.md").decode())
            self.assertEqual(
                archive.read("pwm_modes.parquet"),
                (self.database / "pwm_modes.parquet").read_bytes(),
            )
        for collection, count in [("primary", 2), ("all", 3)]:
            name = f"zebramotif_{collection}_modes_meme.zip"
            with zipfile.ZipFile(self.output / name) as archive:
                self.verify_archive_checksums(archive)
                self.assertEqual(
                    len(
                        [name for name in archive.namelist() if name.endswith(".meme")]
                    ),
                    count,
                )
                self.assertIn("metadata.tsv", archive.namelist())
                self.assertIn("README.md", archive.namelist())
                self.assertNotIn("dataset_README.md", archive.namelist())
                self.assertIn("notices/AF3_OUTPUT_TERMS_OF_USE.md", archive.namelist())
                self.assertFalse(
                    any(
                        name.endswith(".parquet") or name.startswith("structures/")
                        for name in archive.namelist()
                    )
                )
                self.assertEqual(
                    archive.read("metadata.tsv"),
                    (raw / f"zebramotif_{collection}_modes.metadata.tsv").read_bytes(),
                )
            sidecar = (self.output / (name + ".sha256")).read_text().split("  ")[0]
            self.assertEqual(
                sidecar, hashlib.sha256((self.output / name).read_bytes()).hexdigest()
            )
        notices = json.loads(
            (self.export_root / "release/notices/sources.json").read_text()
        )
        for document in notices["documents"]:
            self.assertEqual(
                hashlib.sha256(
                    (raw / "notices" / document["filename"]).read_bytes()
                ).hexdigest(),
                document["sha256"],
            )
        provenance = json.loads((raw / "PROVENANCE.json").read_text())
        self.assertEqual(set(provenance["repositories"]), {"tf-motif-screen"})
        sources = {row["path"]: row for row in provenance["observed_source_files"]}
        self.assertEqual(
            {row["repository"] for row in sources.values()}, {"tf-motif-screen"}
        )
        self.assertEqual(
            sources["LICENSE"]["sha256"],
            hashlib.sha256((self.root / "LICENSE").read_bytes()).hexdigest(),
        )

    def test_zenodo_metadata_preserves_component_terms_under_other_license(self):
        self.export()
        with zipfile.ZipFile(self.output / "zebramotif-data-0.1.zip") as archive:
            metadata = json.loads(archive.read("ZENODO_METADATA.json"))["metadata"]
            rights = metadata["rights"]
            self.assertEqual(len(rights), 1)
            self.assertNotIn("id", rights[0])
            self.assertEqual(rights[0]["title"]["en"], "Other")
            self.assertIn("OUTPUT_TERMS_OF_USE.md", rights[0]["link"])
            description = rights[0]["description"]["en"]
            self.assertIn("LICENSE.md", description)
            self.assertIn("reference annotations", description.lower())
            license_text = archive.read("LICENSE.md").decode()
            self.assertIn("PROSITE", license_text)
            self.assertIn("CC BY-NC-ND 4.0", license_text)
            notice = archive.read("notices/ZebraMotif-MIT.txt").decode()
            self.assertIn("CZ Biohub SF, LLC", notice)

    def test_lookup_and_meme_labels_are_reconciled_and_tsv_retains_unresolved_queries(
        self,
    ):
        self.export()
        raw = self.output / "raw"
        queries = pd.read_parquet(raw / "tf_to_rep.parquet").set_index("DBID")
        self.assertEqual(queries.at[self.gene, "Rep_DBID"], self.gene)
        self.assertTrue(queries.at[self.gene, "is_representative"])
        for collection in ["primary", "all"]:
            text = (raw / f"zebramotif_{collection}_modes.meme").read_text()
            self.assertIn("MOTIF ZM00000010001.1 alpha,member\n", text)
            metadata = pd.read_csv(
                io.StringIO(
                    (raw / f"zebramotif_{collection}_modes.metadata.tsv").read_text()
                ),
                sep="\t",
                dtype={"release_version": str},
            )
            self.assertEqual(set(metadata.query_dbid), set(self.queries.DBID))
            unresolved = metadata.loc[metadata.query_dbid.eq("UNRESOLVED")]
            self.assertEqual(len(unresolved), 1)
            self.assertTrue(unresolved.motif_id.isna().all())

    def test_repeated_exports_are_identical_and_check_does_not_write(self):
        self.export()
        before = inventory(self.output)
        timestamps = {
            path: path.stat().st_mtime_ns
            for path in self.output.rglob("*")
            if path.is_file()
        }
        self.export(check=True)
        self.export()
        self.assertEqual(inventory(self.output), before)
        self.assertEqual(
            {path: path.stat().st_mtime_ns for path in timestamps}, timestamps
        )

    def test_check_detects_corrupted_archives_without_repairing_them(self):
        self.export()
        archive = self.output / "zebramotif-data-0.1.zip"
        archive.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(ValueError, "Release is stale"):
            self.export(check=True)
        self.assertEqual(archive.read_bytes(), b"corrupt archive")

    def test_invalid_notice_does_not_replace_a_completed_release(self):
        self.export()
        before = inventory(self.output)
        (self.export_root / "release/notices/AF3_OUTPUT_TERMS_OF_USE.md").write_text(
            "changed notice"
        )
        with self.assertRaisesRegex(ValueError, "notice changed"):
            self.export()
        self.assertEqual(inventory(self.output), before)

    def test_unmanaged_output_files_are_preserved(self):
        self.output.mkdir(parents=True)
        personal = self.output / "keep.txt"
        personal.write_text("Keep this\n")
        with self.assertRaisesRegex(ValueError, "not a managed release"):
            self.export()
        self.assertEqual(personal.read_text(), "Keep this\n")
        manifest = {
            "schema_version": 1,
            "producer": "another-project/scripts/export_data.py",
            "version": "0.1",
            "files": [{"path": "keep.txt"}],
        }
        (self.output / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "different producer"):
            self.export()
        self.assertEqual(personal.read_text(), "Keep this\n")

    def test_missing_structure_does_not_publish_partial_outputs(self):
        (self.database / self.modes.iloc[0].structure_path).unlink()
        with self.assertRaisesRegex(ValueError, "structure references"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_default_output_uses_the_database_export_directory(self):
        output = export_release(self.root, self.database, schema="# Fixture schema\n")
        self.assertEqual(output, self.output)

    def test_saved_metadata_from_before_the_repository_rename_can_be_exported(self):
        config = json.loads((self.export_root / "release/metadata.json").read_text())
        pipeline_url = config["repositories"].pop("tf-motif-screen")
        previous = "previous-checkout"
        config["repositories"][previous] = f"https://github.com/czbiohub-sf/{previous}"
        config["repositories"]["reference"] = "https://github.com/czbiohub-sf/reference"
        config["reported_versions"]["basis"] = f"{previous}/README.md"
        annotation_basis = config["observed_annotation_versions"]["basis"]
        config["observed_annotation_versions"]["basis"] = (
            f"{previous}/{annotation_basis}"
        )
        config["description"] = f"The {previous} pipeline documents this release."
        saved = self.root / "saved-metadata.json"
        original = json.dumps(config)
        saved.write_text(original)

        self.export(metadata_path=saved)
        self.export(metadata_path=saved, check=True)

        raw = self.output / "raw"
        metadata = json.loads((raw / "release_metadata.json").read_text())
        self.assertEqual(metadata["repositories"]["tf-motif-screen"], pipeline_url)
        self.assertEqual(
            metadata["repositories"]["ZebraMotif"], config["repositories"]["ZebraMotif"]
        )
        self.assertEqual(
            metadata["repositories"]["reference"], config["repositories"]["reference"]
        )
        self.assertEqual(
            metadata["reported_versions"]["basis"], "tf-motif-screen/README.md"
        )
        self.assertEqual(
            metadata["observed_annotation_versions"]["basis"], annotation_basis
        )
        for name in (
            "README.md",
            "release_metadata.json",
            "PROVENANCE.json",
            "ZENODO_METADATA.json",
        ):
            with self.subTest(name=name):
                self.assertNotIn(previous, (raw / name).read_text())
        self.assertEqual(saved.read_text(), original)

    def test_ambiguous_pipeline_repository_is_reported_before_exporting(self):
        config = json.loads((self.export_root / "release/metadata.json").read_text())
        del config["repositories"]["tf-motif-screen"]
        config["repositories"].update(
            first="https://github.com/czbiohub-sf/first",
            second="https://github.com/czbiohub-sf/second",
        )
        saved = self.root / "saved-metadata.json"
        saved.write_text(json.dumps(config))

        with self.assertRaisesRegex(ValueError, r"repositories\.tf-motif-screen"):
            self.export(metadata_path=saved)
        self.assertFalse(self.output.exists())

    def test_relocated_release_from_the_previous_entry_point_is_upgraded(self):
        self.export()
        manifest_path = self.output / "EXPORT_MANIFEST.json"
        for script in ("scripts/export_data.py", "data/database/export_data.py"):
            with self.subTest(script=script):
                manifest = json.loads(manifest_path.read_text())
                manifest["producer"] = f"previous-checkout/{script}"
                manifest_path.write_text(json.dumps(manifest))
                self.export()
                self.assertEqual(
                    json.loads(manifest_path.read_text())["producer"],
                    "tf-motif-screen/data/database/export_data.py",
                )

    @unittest.skipUnless(
        shutil.which("git"), "Git is needed to inspect checkout provenance"
    )
    def test_provenance_tracks_export_code_changes_and_ignores_generated_files(self):
        (self.root / ".gitignore").write_text("data/database/export/\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Export Test",
                "-c",
                "user.email=export@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                f"core.hooksPath={self.root / 'no-hooks'}",
                "commit",
                "-qm",
                "Fixture exporter",
            ],
            check=True,
        )
        self.export()
        path = self.output / "raw/PROVENANCE.json"
        before = json.loads(path.read_text())
        self.assertFalse(before["repositories"]["tf-motif-screen"]["has_local_changes"])
        code = self.export_root / "export_data.py"
        code.write_text("# Changed export code\n")
        self.export()
        after = json.loads(path.read_text())
        self.assertTrue(after["repositories"]["tf-motif-screen"]["has_local_changes"])
        hashes = {row["path"]: row["sha256"] for row in after["observed_source_files"]}
        self.assertEqual(
            hashes["data/database/export_data.py"],
            hashlib.sha256(code.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
