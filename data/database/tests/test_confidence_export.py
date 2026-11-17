import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from export_tools.confidence_export import METRICS, export_confidence_metrics


class ConfidenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.jobdb = self.root / "jobdb.sqlite"
        runs = []
        for index in range(50):
            row = {metric: float(index) for metric in METRICS}
            row.update(protein="modeled", ipTM=(index + 1) / 50, dom_pLDDT=30 + index)
            runs.append(row)
        runs.append({"protein": "extra", **{metric: 999.0 for metric in METRICS}})
        with sqlite3.connect(self.jobdb) as connection:
            pd.DataFrame(runs).to_sql("runs", connection, index=False)

    def test_existing_selection_and_summary_statistics_are_preserved(self):
        before = hashlib.sha256(self.jobdb.read_bytes()).hexdigest()
        result = export_confidence_metrics(
            self.jobdb, self.root / "database", {"MODELED"}
        )
        self.assertEqual(set(result.DBID), {"MODELED"})
        self.assertEqual(len(result), 2 * len(METRICS))
        selected = result.set_index(["subset", "metric"])
        self.assertEqual(selected.loc[("all", "ipTM"), "n"], 50)
        self.assertEqual(selected.loc[("top47", "ipTM"), "n"], 47)
        self.assertEqual(selected.loc[("all", "ipTM"), "mean"], 0.51)
        self.assertEqual(selected.loc[("top47", "ipTM"), "mean"], 0.54)
        self.assertEqual(selected.loc[("top47", "dom_pLDDT"), "p0"], 33.0)
        self.assertEqual(selected.loc[("top47", "dom_pLDDT"), "p100"], 79.0)
        self.assertEqual(hashlib.sha256(self.jobdb.read_bytes()).hexdigest(), before)
        pd.testing.assert_frame_equal(
            pd.read_parquet(self.root / "database/confidence_metrics.parquet"), result
        )

    def test_missing_input_does_not_create_an_empty_job_database(self):
        missing = self.root / "missing.sqlite"
        with self.assertRaises(FileNotFoundError):
            export_confidence_metrics(missing, self.root / "database")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
