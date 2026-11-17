import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from export_tools.database_export import (
    add_sequence_types,
    meme_documents,
    refresh_database_metadata,
    validate_catalog_coverage,
)


class DatabaseExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.full_id = "ENSDARG00000054304"
        self.domain_id = self.full_id + "_2"
        self.supplemental_id = "ENSDARG00000099999"
        self.lookup = pd.DataFrame(
            {
                "DBID": [self.full_id, self.domain_id, self.supplemental_id],
                "TF_Name": ["homeza", "homeza_domain2", "extra_tf"],
                "Rep_DBID": [self.full_id, self.domain_id, self.supplemental_id],
                "is_representative": [True] * 3,
                "resolution": ["self"] * 3,
            }
        )
        motif_ids = ["ZM00000054304.1", "ZM000000543042.1", "ZM00000099999.1"]
        self.modes = pd.DataFrame(
            {
                "DBID": [self.full_id, self.domain_id, self.supplemental_id],
                "TF_Name": ["homeza", "homeza_domain2", "extra_tf"],
                "motif_id": motif_ids,
                "tier": ["primary"] * 3,
                "mode_rank": [1] * 3,
                "n_members": [5, 7, 9],
                "pwm": [[[0.1, 0.2, 0.3, 0.4]]] * 3,
                "structure_path": [
                    "structures/" + name + ".cif.gz" for name in motif_ids
                ],
            }
        )
        self.modes.to_parquet(self.root / "pwm_modes.parquet", index=False)
        self.lookup.to_parquet(self.root / "tf_to_rep.parquet", index=False)
        for name in [
            "interface_residues.parquet",
            "confidence_metrics.parquet",
            "zfin_ids.parquet",
        ]:
            self.lookup[["DBID"]].to_parquet(self.root / name, index=False)
        for name in ["dbd_domains.csv", "domain_annotations.csv"]:
            self.lookup[["DBID"]].to_csv(self.root / name, index=False)
        (self.root / "structures").mkdir()
        for relative in self.modes["structure_path"]:
            (self.root / relative).write_bytes(b"structure")
        self.meme = "MEME version 4\n\n" + "".join(
            f"MOTIF {motif_id}\n\nletter-probability matrix: alength= 4 w= 1\n"
            "0.1 0.2 0.3 0.4\n\n"
            for motif_id in self.modes["motif_id"]
        )
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            (self.root / name).write_text(self.meme)

    def test_refresh_preserves_supplemental_predictions_and_artifacts(self):
        summary = refresh_database_metadata(self.root)
        self.assertEqual(summary["full_length_entries"], 2)
        self.assertEqual(summary["domain_entries"], 1)
        self.assertEqual(summary["removed_modes"], 0)
        actual = pd.read_parquet(self.root / "pwm_modes.parquet")
        expected = self.modes
        pd.testing.assert_frame_equal(actual[self.modes.columns], expected)
        self.assertEqual(
            actual["sequence_type"].tolist(), ["full_length", "domain", "full_length"]
        )
        lookup = pd.read_parquet(self.root / "tf_to_rep.parquet")
        self.assertEqual(
            lookup["sequence_type"].tolist(), ["full_length", "domain", "full_length"]
        )
        for name in [
            "interface_residues.parquet",
            "confidence_metrics.parquet",
            "zfin_ids.parquet",
        ]:
            self.assertIn(
                self.supplemental_id, set(pd.read_parquet(self.root / name)["DBID"])
            )
        for name in ["dbd_domains.csv", "domain_annotations.csv"]:
            self.assertIn(
                self.supplemental_id, set(pd.read_csv(self.root / name)["DBID"])
            )
        for relative in expected["structure_path"]:
            self.assertEqual((self.root / relative).read_bytes(), b"structure")
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            text = (self.root / name).read_text()
            self.assertIn("MOTIF ZM00000099999.1 extra_tf\n", text)
            self.assertIn(
                "w= 1 nsites= 9 E= 0\n0.100000  0.200000  0.300000  0.400000", text
            )
            self.assertEqual(len(re.findall(r"(?m)^MOTIF ", text)), 3)

    def test_refresh_can_be_repeated_without_changing_the_release(self):
        refresh_database_metadata(self.root)
        first = pd.read_parquet(self.root / "pwm_modes.parquet")
        first_files = {
            path.name: path.read_bytes()
            for path in self.root.glob("*")
            if path.is_file()
        }
        summary = refresh_database_metadata(self.root)
        pd.testing.assert_frame_equal(
            pd.read_parquet(self.root / "pwm_modes.parquet"), first
        )
        self.assertEqual(summary["removed_modes"], 0)
        self.assertEqual(
            {
                path.name: path.read_bytes()
                for path in self.root.glob("*")
                if path.is_file()
            },
            first_files,
        )

    def test_refresh_removes_excluded_artifacts_and_keeps_included_gene(self):
        additions = self.modes.iloc[[0, 0]].copy()
        additions["DBID"] = ["ENSDARG00000054778", "ENSDARG00000052618"]
        additions["TF_Name"] = ["excluded_gene", "included_gene"]
        additions["motif_id"] = ["ZM00000054778.1", "ZM00000052618.1"]
        additions["structure_path"] = additions["motif_id"].map(
            lambda name: f"structures/{name}.cif.gz"
        )
        pd.concat([self.modes, additions], ignore_index=True).to_parquet(
            self.root / "pwm_modes.parquet", index=False
        )
        extra_lookup = additions[["DBID", "TF_Name"]].copy()
        extra_lookup["Rep_DBID"] = extra_lookup["DBID"]
        extra_lookup["is_representative"] = True
        extra_lookup["resolution"] = "self"
        pd.concat([self.lookup, extra_lookup], ignore_index=True).to_parquet(
            self.root / "tf_to_rep.parquet", index=False
        )
        for relative in additions["structure_path"]:
            (self.root / relative).write_bytes(b"structure")
        for name in [
            "interface_residues.parquet",
            "confidence_metrics.parquet",
            "dbd_domains.parquet",
            "zfin_ids.parquet",
        ]:
            pd.concat(
                [self.lookup[["DBID"]], extra_lookup[["DBID"]]], ignore_index=True
            ).to_parquet(self.root / name, index=False)

        summary = refresh_database_metadata(self.root)
        self.assertEqual(summary["removed_modes"], 1)
        for name in [
            "pwm_modes.parquet",
            "tf_to_rep.parquet",
            "interface_residues.parquet",
            "confidence_metrics.parquet",
            "dbd_domains.parquet",
            "zfin_ids.parquet",
        ]:
            ids = set(pd.read_parquet(self.root / name)["DBID"])
            self.assertNotIn("ENSDARG00000054778", ids)
            self.assertIn("ENSDARG00000052618", ids)
        self.assertFalse((self.root / additions.iloc[0].structure_path).exists())
        self.assertTrue((self.root / additions.iloc[1].structure_path).exists())
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            text = (self.root / name).read_text()
            self.assertNotIn("excluded_gene", text)
            self.assertIn("MOTIF ZM00000052618.1 included_gene\n", text)

    def test_refresh_repairs_modeled_self_resolution_and_preserves_other_assignments(
        self,
    ):
        add_sequence_types(self.modes).to_parquet(
            self.root / "pwm_modes.parquet", index=False
        )
        self.lookup.loc[0, ["Rep_DBID", "is_representative", "resolution"]] = [
            self.domain_id,
            False,
            "cluster",
        ]
        members = pd.DataFrame(
            {
                "DBID": ["MEMBER", "UNRESOLVED"],
                "TF_Name": ["member", "unresolved"],
                "Rep_DBID": [self.domain_id, None],
                "is_representative": [False, False],
                "resolution": ["cluster", "unresolved"],
            }
        )
        before = pd.concat([self.lookup, members], ignore_index=True)
        before.to_parquet(self.root / "tf_to_rep.parquet", index=False)
        original_modes = (self.root / "pwm_modes.parquet").read_bytes()
        refresh_database_metadata(self.root)
        lookup = pd.read_parquet(self.root / "tf_to_rep.parquet").set_index("DBID")
        self.assertEqual(lookup.at[self.full_id, "Rep_DBID"], self.full_id)
        self.assertTrue(lookup.at[self.full_id, "is_representative"])
        self.assertEqual(lookup.at[self.full_id, "resolution"], "self")
        pd.testing.assert_frame_equal(
            lookup.loc[members.DBID, members.columns[1:]], members.set_index("DBID")
        )
        modes = pd.read_parquet(self.root / "pwm_modes.parquet")
        pd.testing.assert_frame_equal(modes[self.modes.columns], self.modes)
        self.assertEqual((self.root / "pwm_modes.parquet").read_bytes(), original_modes)
        text = (self.root / "zebramotif_primary_modes.meme").read_text()
        self.assertIn("MOTIF ZM00000054304.1 homeza\n", text)
        self.assertIn("MOTIF ZM000000543042.1 homeza_domain2,member\n", text)

    def test_refresh_creates_meme_files_with_sorted_deduplicated_query_symbols(self):
        members = pd.DataFrame(
            {
                "DBID": ["MEMBER1", "MEMBER2", "MEMBER3", "UNNAMED"],
                "TF_Name": ["zeta", "alpha", "alpha", None],
                "Rep_DBID": [self.full_id] * 4,
                "is_representative": [False] * 4,
                "resolution": ["cluster"] * 4,
            }
        )
        pd.concat([self.lookup, members], ignore_index=True).to_parquet(
            self.root / "tf_to_rep.parquet", index=False
        )
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            (self.root / name).unlink()
        refresh_database_metadata(self.root)
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            text = (self.root / name).read_text()
            self.assertIn("MOTIF ZM00000054304.1 alpha,homeza,zeta\n", text)
            self.assertIn("MOTIF ZM000000543042.1 homeza_domain2\n", text)

    def test_meme_collections_derive_matrices_support_and_order_from_tables(self):
        alternate = self.modes.iloc[[0]].copy()
        alternate["tier"] = "alternate"
        alternate["mode_rank"] = 1
        alternate["motif_id"] = "ZM00000054304.2"
        alternate["n_members"] = 3
        alternate["pwm"] = [[[0.4, 0.3, 0.2, 0.1]]]
        modes = pd.concat([alternate, self.modes.iloc[::-1]], ignore_index=True)
        documents = meme_documents(modes, self.lookup)
        self.assertEqual(documents, meme_documents(modes.iloc[::-1], self.lookup))
        primary = documents["zebramotif_primary_modes.meme"]
        all_modes = documents["zebramotif_all_modes.meme"]
        self.assertNotIn("MOTIF ZM00000054304.2 ", primary)
        self.assertLess(
            all_modes.index("MOTIF ZM00000054304.1 "),
            all_modes.index("MOTIF ZM00000054304.2 "),
        )
        self.assertIn(
            "MOTIF ZM00000054304.2 homeza\n\n"
            "letter-probability matrix: alength= 4 w= 1 nsites= 3 E= 0\n"
            "0.400000  0.300000  0.200000  0.100000",
            all_modes,
        )

    def test_refresh_exclusions_remove_motifs_and_structures_from_both_collections(
        self,
    ):
        with patch(
            "export_tools.database_export.EXCLUDED_GENE_IDS", frozenset({self.full_id})
        ):
            summary = refresh_database_metadata(self.root)
        self.assertEqual(summary["removed_modes"], 2)
        for name in ["zebramotif_primary_modes.meme", "zebramotif_all_modes.meme"]:
            text = (self.root / name).read_text()
            self.assertNotIn("MOTIF ZM00000054304", text)
            self.assertIn("MOTIF ZM00000099999.1 extra_tf\n", text)
        for relative in self.modes.loc[
            self.modes.DBID.ne(self.supplemental_id), "structure_path"
        ]:
            self.assertFalse((self.root / relative).exists())

    def test_invalid_meme_matrix_is_rejected_before_any_exports_change(self):
        self.modes.at[0, "pwm"] = [[0.1, 0.2, 0.7]]
        self.modes.to_parquet(self.root / "pwm_modes.parquet", index=False)
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaisesRegex(ValueError, r"\(L, 4\)"):
            refresh_database_metadata(self.root)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_dangling_representative_is_rejected_before_writing(self):
        self.lookup.loc[len(self.lookup)] = [
            "MEMBER",
            "member",
            self.supplemental_id,
            False,
            "cluster",
        ]
        self.lookup.to_parquet(self.root / "tf_to_rep.parquet", index=False)
        original = (self.root / "pwm_modes.parquet").read_bytes()
        with patch(
            "export_tools.database_export.EXCLUDED_GENE_IDS",
            frozenset({self.supplemental_id}),
        ):
            with self.assertRaisesRegex(ValueError, "without primary modes"):
                refresh_database_metadata(self.root)
        self.assertEqual((self.root / "pwm_modes.parquet").read_bytes(), original)

    def test_supplemental_gene_domains_are_retained_without_mutating_the_input(self):
        frame = pd.DataFrame(
            {"DBID": [self.full_id, self.domain_id, self.supplemental_id + "_1"]}
        )
        original = frame.copy()
        result = add_sequence_types(frame)
        self.assertEqual(result["DBID"].tolist(), frame["DBID"].tolist())
        self.assertEqual(
            result["sequence_type"].tolist(), ["full_length", "domain", "domain"]
        )
        pd.testing.assert_frame_equal(frame, original)


class CatalogCoverageTests(unittest.TestCase):
    def test_all_catalog_ids_resolve_with_extra_domain_entries(self):
        lookup = pd.DataFrame(
            {
                "DBID": ["CATALOG_GENE", "MEMBER", "DOMAIN_1"],
                "Rep_DBID": ["CATALOG_GENE", "CATALOG_GENE", "DOMAIN_1"],
            }
        )
        validate_catalog_coverage(
            lookup, pd.DataFrame({"DBID": ["CATALOG_GENE", "MEMBER"]})
        )

    def test_missing_catalog_id_is_rejected(self):
        lookup = pd.DataFrame({"DBID": ["OTHER"], "Rep_DBID": ["OTHER"]})
        with self.assertRaisesRegex(ValueError, "missing query IDs.*CATALOG_GENE"):
            validate_catalog_coverage(lookup, pd.DataFrame({"DBID": ["CATALOG_GENE"]}))

    def test_present_but_unresolved_catalog_id_is_rejected(self):
        lookup = pd.DataFrame({"DBID": ["CATALOG_GENE"], "Rep_DBID": [None]})
        with self.assertRaisesRegex(ValueError, "unresolved query IDs.*CATALOG_GENE"):
            validate_catalog_coverage(lookup, pd.DataFrame({"DBID": ["CATALOG_GENE"]}))


if __name__ == "__main__":
    unittest.main()
