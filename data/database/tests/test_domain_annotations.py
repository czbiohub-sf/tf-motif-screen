import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "export_data.py"
SPEC = importlib.util.spec_from_file_location("export_data", SCRIPT)
database = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(database)


class DomainAnnotationExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "domains.csv"
        self.domains = pd.DataFrame(
            [
                ["PLACED", "placed_tf", "IPR1", "Domain A", 10, 50, True],
                ["PLACED", "placed_tf", "IPR2", "Domain B", 60, 90, False],
                ["UNPLACED", "unplaced_tf", "IPR3", "Domain C", 20, 80, False],
                ["OTHER", "other_tf", "IPR4", "Domain D", 1, 10, True],
            ],
            columns=database.DOMAIN_COLS,
        )
        self.domains.to_csv(self.source, index=False)
        self.keep = {"PLACED", "UNPLACED", "UNANNOTATED"}

    def test_full_export_preserves_annotated_but_unplaced_domains(self):
        result = database.build_domain_annotations(self.source, self.keep)
        self.assertEqual(set(result.DBID), {"PLACED", "UNPLACED"})
        self.assertEqual(len(result), 3)
        self.assertTrue(result.groupby("DBID")["DNA-Binding"].any()["PLACED"])
        self.assertFalse(result.groupby("DBID")["DNA-Binding"].any()["UNPLACED"])
        self.assertNotIn("UNANNOTATED", set(result.DBID))

    def test_highlights_are_the_flagged_subset_of_the_full_export(self):
        database.export_domain_annotations(self.source, self.root, self.keep)
        all_domains = pd.read_parquet(self.root / "dbd_domains.parquet")
        highlights = all_domains.loc[all_domains["DNA-Binding"]]
        self.assertEqual(highlights.DBID.tolist(), ["PLACED"])
        self.assertEqual(len(all_domains), 3)
        self.assertFalse((self.root / "domain_annotations.csv").exists())
        self.assertFalse((self.root / "dbd_domains.csv").exists())

    def test_unknown_flags_are_not_exported_as_no_confident_placement(self):
        self.domains["DNA-Binding"] = self.domains["DNA-Binding"].astype(object)
        for value in (None, "uncomputed"):
            with self.subTest(value=value):
                self.domains.loc[0, "DNA-Binding"] = value
                self.domains.to_csv(self.source, index=False)
                with self.assertRaisesRegex(ValueError, "complete boolean"):
                    database.build_domain_annotations(self.source, self.keep)

    def test_integer_flags_are_not_mistaken_for_row_indices(self):
        self.domains["DNA-Binding"] = self.domains["DNA-Binding"].astype(int)
        self.domains.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "complete boolean"):
            database.build_domain_annotations(self.source, self.keep)

    def test_empty_annotation_table_exports_without_inventing_domains(self):
        self.domains.iloc[:0].to_csv(self.source, index=False)
        database.export_domain_annotations(self.source, self.root, self.keep)
        result = pd.read_parquet(self.root / "dbd_domains.parquet")
        self.assertTrue(result.empty)
        self.assertTrue(pd.api.types.is_bool_dtype(result["DNA-Binding"]))

    def test_missing_source_does_not_imply_no_domains_annotated(self):
        with self.assertRaises(FileNotFoundError):
            database.build_domain_annotations(self.root / "missing.csv", self.keep)


if __name__ == "__main__":
    unittest.main()
