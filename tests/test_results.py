import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results import ratios, recount, roc_auc, score_plasgraph, validate


class ResultsTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(ratios(1, 1, 0), (0.5, 1, 2/3))
        self.assertEqual(ratios(0, 0, 0), (0, 0, 0))

    def test_tied_roc_auc(self):
        self.assertEqual(roc_auc([1, 0], [0.5, 0.5]), 0.5)
        self.assertIsNone(roc_auc([1], [0.9]))

    def test_plasgraph_gold(self):
        gold = [
            dict(sample="a", contig="1", length="1000", label="ambiguous"),
            dict(sample="a", contig="2", length="100", label="chromosome"),
            dict(sample="a", contig="3", length="1000", label="unlabeled"),
        ]
        predictions = [
            dict(sample="a", contig="1", plasmid_score="0.8", chrom_score="0.9"),
            dict(sample="a", contig="2", plasmid_score="0.5", chrom_score="0.9"),
        ]
        plasmid, chrom, long_plasmid, _ = score_plasgraph(gold, predictions)
        self.assertEqual((plasmid["real"], chrom["real"], long_plasmid["contigs"]), (1, 2, 1))
        self.assertEqual(plasmid["f1"], 1.0)

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
