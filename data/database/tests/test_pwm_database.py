import importlib.util
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "export_data.py"
SPEC = importlib.util.spec_from_file_location("export_data", SCRIPT)
database = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(database)


class LookupExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        data_patch = patch.object(database, "DATA", self.data)
        self.addCleanup(data_patch.stop)
        data_patch.start()
        database_patch = patch.object(
            database, "DEFAULT_DATABASE", self.data / "database"
        )
        self.addCleanup(database_patch.stop)
        database_patch.start()
        release_patch = patch.object(database, "export_release")
        self.addCleanup(release_patch.stop)
        self.release_export = release_patch.start()
        self.first = "ENSDARG00000010001"
        self.second = "ENSDARG00000010002"
        self.failed = "ENSDARG00000010003"
        self.member = "ENSDARG00000010004"
        self.failed_member = "ENSDARG00000010005"
        self.reps = pd.DataFrame(
            {
                "DBID": [self.first, self.second, self.failed],
                "TF_Name": ["alpha", "beta", "failed"],
                "Family": ["family"] * 3,
                "Hom_JASPAR": [None] * 3,
                "Hom_Sim": [None] * 3,
            }
        )
        self.full = pd.concat(
            [
                self.reps,
                pd.DataFrame(
                    {
                        "DBID": [self.member, self.failed_member],
                        "TF_Name": ["member", "failed_member"],
                        "Family": ["family"] * 2,
                    }
                ),
            ],
            ignore_index=True,
        )
        clusters = self.data / "tools/mmseqs/animaltfdb4/results"
        clusters.mkdir(parents=True)
        (clusters / "family.mmseqs_cluster.tsv").write_text(
            f"{self.second}\t{self.first}\n"
            f"{self.second}\t{self.member}\n"
            f"{self.failed}\t{self.failed_member}\n"
        )
        zfin = self.data / database.ZFIN_REFERENCE
        zfin.parent.mkdir(parents=True)
        pd.DataFrame({"DBID": [self.first], "zfin_id": ["ZDB-GENE-1"]}).to_parquet(
            zfin, index=False
        )

    def test_modeled_query_resolves_to_itself_despite_cluster_redirect(self):
        lookup = database.build_tf_to_rep(self.reps, self.full).set_index("DBID")
        self.assertEqual(lookup.at[self.first, "Rep_DBID"], self.first)
        self.assertTrue(lookup.at[self.first, "is_representative"])
        self.assertEqual(lookup.at[self.first, "resolution"], "self")
        self.assertEqual(lookup.at[self.member, "Rep_DBID"], self.second)

    def test_missing_predictions_remain_queryable_without_dangling_references(self):
        lookup = database.build_tf_to_rep(
            self.reps, self.full, {self.first, self.second}
        ).set_index("DBID")
        for dbid in [self.failed, self.failed_member]:
            self.assertIn(dbid, lookup.index)
            self.assertTrue(pd.isna(lookup.at[dbid, "Rep_DBID"]))
            self.assertFalse(lookup.at[dbid, "is_representative"])
            self.assertEqual(lookup.at[dbid, "resolution"], "unresolved")
        self.assertEqual(lookup.at[self.member, "Rep_DBID"], self.second)

    def mode(self):
        return {
            "DBID": self.first,
            "TF_Name": "alpha",
            "motif_id": "ZM00000010001.1",
            "tier": "primary",
            "mode_rank": 1,
            "n_members": 20,
            "num_specs": 47,
            "pwm": [[0.1, 0.2, 0.3, 0.4]],
            "structure_path": None,
            "homer_sim": None,
            "recovery": None,
        }

    def test_lookup_only_repairs_without_loading_research_annotations(self):
        out = self.data / "database"
        out.mkdir()
        pd.DataFrame([self.mode()]).to_parquet(out / "pwm_modes.parquet", index=False)
        lookup = database.build_tf_to_rep(self.reps, self.full, {self.first})
        lookup.loc[
            lookup.DBID.eq(self.first), ["Rep_DBID", "is_representative", "resolution"]
        ] = [self.second, False, "cluster"]
        lookup.to_parquet(out / "tf_to_rep.parquet", index=False)
        with (
            patch("sys.argv", [str(SCRIPT), "--lookup-only"]),
            patch.object(
                database,
                "load_zfin_ids",
                side_effect=AssertionError("Research input read"),
            ),
        ):
            database.main()
        actual = pd.read_parquet(out / "tf_to_rep.parquet").set_index("DBID")
        self.assertEqual(actual.at[self.first, "Rep_DBID"], self.first)
        self.assertEqual(actual.at[self.first, "resolution"], "self")
        self.assertTrue(actual.at[self.first, "is_representative"])
        for collection in ["primary", "all"]:
            text = (out / f"zebramotif_{collection}_modes.meme").read_text()
            self.assertIn("MOTIF ZM00000010001.1 alpha\n", text)

    def test_full_build_finalizes_lookup_and_writes_meme_after_prediction_results(self):
        sequences = self.data / "sequences/animaltfdb"
        sequences.mkdir(parents=True)
        self.reps.to_csv(sequences / "tfdb4_rep_tfs.csv", index=False)
        self.full.to_csv(sequences / "tfdb4_tfs.csv", index=False)
        pd.DataFrame(
            [[self.first, "alpha", "IPR1", "Domain", 1, 10, True]],
            columns=database.DOMAIN_COLS,
        ).to_csv(sequences / "tfdb4_rep_doms.csv", index=False)

        def completed_prediction(_function, row, *_args):
            future = Future()
            future.set_result([self.mode()] if row["DBID"] == self.first else [])
            return future

        # Exercise the export orchestration while replacing expensive prediction
        # extraction and the unrelated JASPAR reader with small completed results.
        motif_reader = SimpleNamespace(
            load_homer_multi_motif_file=lambda _path: ([], [])
        )
        with (
            patch.dict("sys.modules", {"af3tfm.motif": motif_reader}),
            patch.object(database, "ProcessPoolExecutor") as executor,
            patch.object(database, "export_confidence_metrics") as confidence_export,
            patch("sys.argv", [str(SCRIPT)]),
        ):
            executor.return_value.__enter__.return_value.submit.side_effect = (
                completed_prediction
            )
            database.main()
            confidence_export.assert_called_once_with(
                self.data / "jobs/hero/jobdb.sqlite",
                self.data / "database",
                {self.first},
                47,
            )
            self.release_export.assert_called_once()

        out = self.data / "database"
        lookup = pd.read_parquet(out / "tf_to_rep.parquet").set_index("DBID")
        self.assertEqual(lookup.at[self.first, "Rep_DBID"], self.first)
        self.assertTrue(lookup.at[self.first, "is_representative"])
        self.assertTrue(lookup.loc[lookup.index != self.first, "Rep_DBID"].isna().all())
        for collection in ["primary", "all"]:
            text = (out / f"zebramotif_{collection}_modes.meme").read_text()
            self.assertIn("MOTIF ZM00000010001.1 alpha\n", text)
            self.assertEqual(text.count("\nMOTIF "), 1)
            self.assertIn("w= 1 nsites= 20 E= 0", text)

    def test_from_existing_and_check_never_rebuild_the_scientific_database(self):
        for arguments in [["--from-existing"], ["--check"]]:
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", [str(SCRIPT), *arguments]),
                patch.object(
                    database,
                    "build_database",
                    side_effect=AssertionError("Rebuilt motifs"),
                ),
                patch.object(
                    database,
                    "export_confidence_metrics",
                    side_effect=AssertionError("Rebuilt confidence"),
                ),
            ):
                self.release_export.reset_mock()
                database.main()
                self.release_export.assert_called_once()
                self.assertEqual(
                    self.release_export.call_args.kwargs["check"],
                    "--check" in arguments,
                )


if __name__ == "__main__":
    unittest.main()
