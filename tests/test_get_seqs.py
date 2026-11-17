import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "data/sequences/animaltfdb/get_seqs.py"
SPEC = importlib.util.spec_from_file_location("animaltfdb_get_seqs", SCRIPT)
animaltfdb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(animaltfdb)
ensembl = sys.modules["get_seqs"]


class EnsemblFetchTests(unittest.TestCase):
    def test_batches_protein_ids_and_retains_only_returned_records(self):
        ids = [f"ENSP{i:011d}" for i in range(51)]
        first = [{"id": pid, "version": 2, "seq": "MKK"} for pid in ids[:49]]
        last = [{"id": ids[-1], "seq": "MNN"}]
        with (
            patch.object(ensembl, "_post_json", side_effect=[first, last]) as post,
            redirect_stdout(io.StringIO()),
        ):
            proteins = ensembl.fetch_proteins(ids)

        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].args[1], {"ids": ids[:50]})
        self.assertEqual(post.call_args_list[1].args[1], {"ids": ids[50:]})
        self.assertEqual(proteins[ids[0]], (2, "MKK"))
        self.assertEqual(proteins[ids[-1]], (None, "MNN"))
        self.assertNotIn(ids[49], proteins)

    def test_retries_transient_failures_and_honors_retry_after(self):
        response = [{"id": "ENSP1", "seq": "MKK"}]
        responses = [
            ensembl.RequestException("Temporary connection failure"),
            Mock(status_code=429, headers={"Retry-After": "3"}),
            Mock(status_code=503),
            Mock(status_code=200, json=Mock(return_value=response)),
        ]
        with (
            patch.object(ensembl.requests, "post", side_effect=responses) as post,
            patch.object(ensembl.time, "sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            result = ensembl._post_json("https://example.org", {"ids": ["ENSP1"]})

        self.assertEqual(result, response)
        self.assertEqual(post.call_count, 4)
        self.assertIn(call(3.0), sleep.call_args_list)

    def test_does_not_retry_permanent_client_errors(self):
        with (
            patch.object(
                ensembl.requests, "post", return_value=Mock(status_code=400)
            ) as post,
            patch.object(ensembl.time, "sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            result = ensembl._post_json("https://example.org", {"ids": ["ENSP1"]})

        self.assertIsNone(result)
        post.assert_called_once()
        sleep.assert_not_called()


class AnimalTFDBSequenceTests(unittest.TestCase):
    FASTA = ">ENSP1.1 GENE1 alpha Homeobox\nAAAA\n>ENSP2.1 GENE2 beta Fork_head\nCCCC\n"

    def test_outputs_use_ensembl_sequences_versions_and_existing_csv_schema(self):
        messages = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tfs.fasta"
            with (
                patch.object(animaltfdb, "download_fasta", return_value=self.FASTA),
                patch.object(
                    animaltfdb, "fetch_proteins", return_value={"ENSP1": (2, "MKK")}
                ),
                patch.object(sys, "argv", [str(SCRIPT), "--filename", str(output)]),
                redirect_stdout(messages),
            ):
                animaltfdb.main()

            table = pd.read_csv(output.with_suffix(".csv"))
            self.assertEqual(
                table.columns.tolist(), ["DBID", "TF_Name", "Family", "Sequence"]
            )
            self.assertEqual(table.DBID.tolist(), ["GENE1"])
            self.assertEqual(table.Sequence.tolist(), ["MKK"])
            self.assertEqual(
                output.read_text(), ">ENSP1.2\tGENE1\talpha\tHomeobox\nMKK\n"
            )
        self.assertIn("version 2", messages.getvalue())
        self.assertIn("ENSP2 (beta) not found", messages.getvalue())

    def test_manual_input_uses_saved_headers_without_contacting_animaltfdb(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "downloaded.fasta"
            source.write_text(self.FASTA)
            output = Path(directory) / "tfs.fasta"
            with (
                patch.object(animaltfdb, "download_fasta") as download,
                patch.object(
                    animaltfdb,
                    "fetch_proteins",
                    return_value={"ENSP1": (1, "MKK"), "ENSP2": (1, "MNN")},
                ) as fetch,
                patch.object(
                    sys,
                    "argv",
                    [str(SCRIPT), "--input", str(source), "--filename", str(output)],
                ),
            ):
                animaltfdb.main()
            download.assert_not_called()
            fetch.assert_called_once_with(["ENSP1", "ENSP2"])
            table = pd.read_csv(output.with_suffix(".csv"))
            self.assertEqual(table.Sequence.tolist(), ["MKK", "MNN"])
            self.assertEqual(table.Family.tolist(), ["Homeobox", "Fork_head"])
            self.assertEqual(source.read_text(), self.FASTA)

    def test_browser_challenge_is_rejected_without_retry_or_output(self):
        response = Mock(text="<html>Browser interaction required</html>")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tfs.fasta"
            with (
                patch.object(animaltfdb.requests, "get", return_value=response) as get,
                patch.object(animaltfdb, "fetch_proteins") as fetch,
                patch.object(sys, "argv", [str(SCRIPT), "--filename", str(output)]),
                patch.object(sys, "stderr", new_callable=io.StringIO) as errors,
                self.assertRaises(SystemExit) as raised,
            ):
                animaltfdb.main()
            self.assertEqual(raised.exception.code, 2)
            get.assert_called_once()
            fetch.assert_not_called()
            self.assertIn("--input", errors.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".csv").exists())

    def test_failed_download_is_not_retried(self):
        with patch.object(
            animaltfdb.requests,
            "get",
            side_effect=animaltfdb.requests.HTTPError("403 Forbidden"),
        ) as get:
            with self.assertRaises(animaltfdb.requests.HTTPError):
                animaltfdb.download_fasta("https://example.org/proteins.fasta")
        get.assert_called_once_with("https://example.org/proteins.fasta", timeout=30)

    def test_rejects_empty_or_malformed_catalogs(self):
        for fasta in ["", "<html>Not a FASTA file</html>", ">ENSP1\nAAAA\n"]:
            with self.subTest(fasta=fasta), self.assertRaises(ValueError):
                animaltfdb.parse_fasta(fasta)

    def test_fasta_omits_an_unknown_ensembl_version(self):
        table = animaltfdb.parse_fasta(self.FASTA)
        with (
            patch.object(
                animaltfdb, "fetch_proteins", return_value={"ENSP1": (None, "MKK")}
            ),
            redirect_stdout(io.StringIO()),
        ):
            table = animaltfdb.add_sequences(table)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tfs.fasta"
            animaltfdb.write_fasta(table, output)
            self.assertEqual(
                output.read_text(), ">ENSP1\tGENE1\talpha\tHomeobox\nMKK\n"
            )


if __name__ == "__main__":
    unittest.main()
