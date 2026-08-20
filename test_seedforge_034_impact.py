import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import seedforge_034_impact as impact


class ImpactSelectionTests(unittest.TestCase):
    def make_frame(self):
        fixed = []
        for number, spec in enumerate(impact.FIXED_CANDIDATES):
            fixed.append({
                "job_id": f"fixed-{number}",
                "factor_combo": spec["factor_combo"],
                "portfolio_size": spec["portfolio_size"],
                "rebalance_months": spec["rebalance_months"],
                "exit_rank_multiplier": spec["exit_rank_multiplier"],
                "weight_policy": spec["weight_policy"],
                "train_selection_score": number + 1.0,
                "turnover": number + 10.0,
            })
        extras = [{
            "job_id": f"extra-{number}", "factor_combo": f"X{number}",
            "portfolio_size": 20, "rebalance_months": 1,
            "exit_rank_multiplier": 1.0, "weight_policy": "equal",
            "train_selection_score": float(number), "turnover": float(number),
        } for number in range(100)]
        return pd.DataFrame([*fixed, *extras])

    def test_selects_six_unique_candidates_without_opened_columns(self):
        selected = impact.select_fixed_candidates(self.make_frame(), 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected["job_id"].nunique(), 6)
        self.assertFalse(any("opened" in column for column in selected.columns))
        self.assertEqual(
            selected["selection_reason"].tolist()[:3],
            [spec["selection_reason"] for spec in impact.FIXED_CANDIDATES],
        )

    def test_legacy_hash_check_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for number in range(4):
                path = Path(directory) / f"legacy-{number}.txt"
                path.write_text("before", encoding="utf-8")
                paths.append(path)
            with patch.object(impact, "LEGACY_FILES", tuple(paths)), patch.object(
                impact, "ROOT", Path(directory)
            ):
                before = impact.legacy_hashes()
                paths[0].write_text("after", encoding="utf-8")
                self.assertNotEqual(before, impact.legacy_hashes())


if __name__ == "__main__":
    unittest.main()

