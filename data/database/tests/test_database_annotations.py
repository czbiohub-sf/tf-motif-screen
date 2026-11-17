import tempfile
import unittest
from pathlib import Path

import pandas as pd

from export_tools.database_export import (
    DOMAIN_COLUMNS,
    add_lookup_annotations,
    consolidate_database_annotations,
    load_zfin_ids,
)


class DatabaseAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gene = "ENSDARG00000054304"
        self.domain = self.gene + "_2"
        self.unannotated = "ENSDARG00000009473"
        self.lookup = pd.DataFrame({"DBID": [self.gene, self.domain, self.unannotated]})
        self.lookup.to_parquet(self.root / "tf_to_rep.parquet", index=False)
        self.lookup.to_parquet(self.root / "pwm_modes.parquet", index=False)
        self.catalog = pd.DataFrame({"DBID": [self.gene, self.unannotated]})
        self.zfin = pd.DataFrame(
            {
                "DBID": [self.gene, self.unannotated],
                "zfin_id": ["ZDB-GENE-1", "ZDB-GENE-2"],
            }
        )
        self.domains = pd.DataFrame(
            [
                [self.gene, "homeza", "IPR1", "Domain A", 10, 50, True],
                [self.gene, "homeza", "IPR2", "Domain B", 60, 90, False],
                [self.domain, "homeza_domain2", "IPR2", "Domain B", 1, 30, False],
                ["OTHER", "other_tf", "IPR3", "Domain C", 1, 20, True],
            ],
            columns=DOMAIN_COLUMNS,
        )
        self.domains.to_csv(self.root / "domain_annotations.csv", index=False)
        self.domains.loc[self.domains["DNA-Binding"]].to_csv(
            self.root / "dbd_domains.csv", index=False
        )
        self.zfin.to_parquet(self.root / "zfin_ids.parquet", index=False)
        (self.root / "animaltfdb4_tf_ids.txt").write_text(self.gene + "\n")

    def test_consolidation_preserves_domains_and_keeps_query_metadata_distinct(self):
        modes_before = (self.root / "pwm_modes.parquet").read_bytes()
        summary = consolidate_database_annotations(
            self.root, self.catalog, self.zfin, self.domains
        )
        lookup = pd.read_parquet(self.root / "tf_to_rep.parquet").set_index("DBID")
        self.assertTrue(lookup.loc[self.gene, "in_original_catalog"])
        self.assertFalse(lookup.loc[self.domain, "in_original_catalog"])
        self.assertEqual(lookup.loc[self.domain, "sequence_type"], "domain")
        self.assertTrue(pd.isna(lookup.loc[self.domain, "zfin_id"]))
        self.assertEqual(lookup.loc[self.gene, "zfin_id"], "ZDB-GENE-1")
        domains = pd.read_parquet(self.root / "dbd_domains.parquet")
        pd.testing.assert_frame_equal(
            domains, self.domains.iloc[:3].reset_index(drop=True)
        )
        self.assertNotIn(self.unannotated, set(domains.DBID))
        self.assertFalse(domains.loc[domains.DBID.eq(self.domain), "DNA-Binding"].any())
        self.assertEqual(summary["original_catalog_entries"], 2)
        self.assertEqual(summary["zfin_links"], 2)
        self.assertEqual((self.root / "pwm_modes.parquet").read_bytes(), modes_before)
        for name in [
            "domain_annotations.csv",
            "dbd_domains.csv",
            "zfin_ids.parquet",
            "animaltfdb4_tf_ids.txt",
        ]:
            self.assertFalse((self.root / name).exists())

    def test_consolidation_is_repeatable_without_legacy_sidecars(self):
        consolidate_database_annotations(
            self.root, self.catalog, self.zfin, self.domains
        )
        first = pd.read_parquet(self.root / "tf_to_rep.parquet")
        consolidate_database_annotations(
            self.root, self.catalog, self.zfin, self.domains
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(self.root / "tf_to_rep.parquet"), first
        )

    def test_invalid_domain_flags_leave_existing_exports_untouched(self):
        original = (self.root / "tf_to_rep.parquet").read_bytes()
        self.domains["DNA-Binding"] = self.domains["DNA-Binding"].astype(int)
        with self.assertRaisesRegex(ValueError, "complete boolean"):
            consolidate_database_annotations(
                self.root, self.catalog, self.zfin, self.domains
            )
        self.assertEqual((self.root / "tf_to_rep.parquet").read_bytes(), original)
        self.assertTrue((self.root / "domain_annotations.csv").exists())

    def test_conflicting_zfin_mappings_are_rejected(self):
        conflicting = pd.concat(
            [
                self.zfin,
                pd.DataFrame({"DBID": [self.gene], "zfin_id": ["ZDB-GENE-OTHER"]}),
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicting IDs"):
            add_lookup_annotations(self.lookup, self.catalog, conflicting)

    def test_frozen_mapping_loads_from_parquet_without_a_text_download(self):
        path = self.root / "frozen_zfin.parquet"
        self.zfin.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(load_zfin_ids(path), self.zfin)


if __name__ == "__main__":
    unittest.main()
