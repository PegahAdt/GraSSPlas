import csv
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_training_and_classification_resolve_cohort_graphs(self):
        run = runpy.run_path(str(ROOT / "experiments/weak-supervision/run.py"))["run"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            adaptation = root / "lab adapter"
            data = adaptation / "data"
            cohort = adaptation / "sets/plaseval-nolabels.csv"
            inputs = root / "inputs"
            source = adaptation / "src"
            config = root / "model.yaml"
            for folder in (cohort.parent, inputs, source):
                folder.mkdir(parents=True)
            cohort.write_text("plaseval/sample/graph.gfa.gz,,sample\n")
            graph = data / "plaseval/sample/graph.gfa.gz"
            graph.parent.mkdir(parents=True)
            graph.touch()
            for name in ("plASgraph2_train.py", "plASgraph2_classify.py", "eval.py"):
                (source / name).touch()
            for name in ("mob2-extfeatures.csv", "mob2-pseudolabels.csv", "mob-ground-truth.csv"):
                (inputs / name).touch()
            config.touch()

            def check_graph_prefix(command, **_kwargs):
                program = Path(command[1]).name
                if program in ("plASgraph2_train.py", "plASgraph2_classify.py"):
                    prefix = command[command.index(str(cohort)) + 1]
                    with cohort.open(newline="") as stream:
                        for relative_path, _, _ in csv.reader(stream):
                            # The adapted reader appends each relative path as a string.
                            self.assertEqual(Path(prefix + relative_path), graph)
                            self.assertTrue(Path(prefix + relative_path).is_file())

            args = SimpleNamespace(adaptation=adaptation, inputs=inputs, condition="mob2",
                                   config=config, output=root / "output")
            with patch("subprocess.run", side_effect=check_graph_prefix) as child:
                run(args, "mob2")
            self.assertEqual(
                [Path(call.args[0][1]).name for call in child.call_args_list],
                ["plASgraph2_train.py", "plASgraph2_classify.py", "eval.py", "eval.py"],
            )


if __name__ == "__main__":
    unittest.main()
