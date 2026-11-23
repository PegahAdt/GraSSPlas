import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_inputs import check_digest
from validate_results import ratios, recount, validate


class PortfolioTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(ratios(1, 1, 0), (0.5, 1, 2/3))
        self.assertEqual(ratios(0, 0, 0), (0, 0, 0))

    def test_input_mismatch(self):
        with self.assertRaises(ValueError):
            check_digest(b"different", "123", "sample")

    def test_order_mismatch(self):
        with TemporaryDirectory() as d:
            path = Path(d)
            for name, content in [("features.csv", "contig,length\na,1000\n"),
                                  ("ground_truth.csv", "contig,Plasmid\nb,1\n"),
                                  ("y_final.csv", "y_final\n1\n")]:
                (path / name).write_text(content)
            with self.assertRaisesRegex(ValueError, "order"):
                recount(path)

    def test_published_results(self):
        validate()


if __name__ == "__main__":
    unittest.main()
