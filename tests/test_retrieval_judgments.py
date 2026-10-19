"""An annotation template must never be mistaken for graded retrieval truth."""
import tempfile
import unittest
from pathlib import Path

from jev_tmcad.study_retrieval import load_judgments


class JudgmentLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "retrieval_judgments.csv"

    def test_blank_template_reports_count_and_location(self):
        self.path.write_text("split,query_id,candidate_id,grade\nval,a,b,\ntest,c,d,\n")
        with self.assertRaisesRegex(ValueError, "2 blank and 0 invalid grades among 2 pairs"):
            load_judgments(self.path)

    def test_invalid_grade_is_rejected(self):
        self.path.write_text("split,query_id,candidate_id,grade\nval,a,b,1.5\n")
        with self.assertRaisesRegex(ValueError, "0 blank and 1 invalid"):
            load_judgments(self.path)

    def test_valid_grades_remain_integer_and_ids_remain_strings(self):
        self.path.write_text("split,query_id,candidate_id,grade\nval,001,002,0\ntest,003,004,3\n")
        frame = load_judgments(self.path)
        self.assertEqual(frame["grade"].tolist(), [0, 3])
        self.assertEqual(frame["query_id"].tolist(), ["001", "003"])


if __name__ == "__main__":
    unittest.main()
