import json
import tempfile
import unittest
from pathlib import Path

from seedforge_lab_guardrails import (
    append_experiment,
    experiment_fingerprint,
    meaning_fingerprint,
    reproduction_fingerprint,
    sanitize,
    validate_preset,
    validate_rule,
)


INDICATORS = {"CUSTOM_RSI", "RSI_ALL_BEAR_COUNT", "M01_ret20"}


class GuardrailTests(unittest.TestCase):
    def test_rsi_range_requires_period_and_upper(self):
        with self.assertRaisesRegex(ValueError, "기간,상한"):
            validate_rule(
                {"side": "매수", "indicator": "CUSTOM_RSI", "operator": "범위 안", "value": 30, "parameter": "70"},
                INDICATORS,
            )

    def test_count_range_is_normalized(self):
        rule = validate_rule(
            {"side": "매도", "indicator": "RSI_ALL_BEAR_COUNT", "operator": "범위 안", "value": 1, "parameter": "60,3"},
            INDICATORS,
        )
        self.assertEqual(rule["parameter"], "60,3")
        self.assertEqual(rule["value"], 1.0)

    def test_non_finite_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_rule(
                {"side": "매수", "indicator": "M01_ret20", "operator": "이상", "value": "nan"},
                INDICATORS,
            )

    def test_preset_requires_horizon(self):
        with self.assertRaisesRegex(ValueError, "보유기간"):
            validate_preset(
                {
                    "mode": "매수 테스트",
                    "logic": "모두 충족(AND)",
                    "horizons": [],
                    "rules": [{"side": "매수", "indicator": "M01_ret20", "operator": "상위 %", "value": 20}],
                },
                INDICATORS,
            )

    def test_fingerprint_is_order_independent_for_keys(self):
        first = {"mode": "매수 테스트", "logic": "모두 충족(AND)", "rules": [], "horizons": [4]}
        second = {"horizons": [4], "rules": [], "logic": "모두 충족(AND)", "mode": "매수 테스트"}
        self.assertEqual(experiment_fingerprint(first), experiment_fingerprint(second))

    def test_meaning_fingerprint_ignores_build(self):
        base = {"mode": "매수 테스트", "logic": "모두 충족(AND)", "rules": [], "horizons": [4]}
        self.assertEqual(
            meaning_fingerprint({**base, "build": "lab-1"}),
            meaning_fingerprint({**base, "build": "lab-9"}),
        )

    def test_reproduction_fingerprint_tracks_build_and_data(self):
        spec = {"mode": "매수 테스트", "logic": "모두 충족(AND)", "rules": [], "horizons": [4], "build": "lab-9"}
        self.assertNotEqual(
            reproduction_fingerprint(spec, "2026-01-01"),
            reproduction_fingerprint(spec, "2026-02-01"),
        )

    def test_sanitize_is_public_and_recursive(self):
        self.assertEqual(sanitize({"x": [float("nan"), float("inf"), 1.0]}), {"x": [None, None, 1.0]})

    def test_ledger_counts_repeated_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = append_experiment(path, {"rule": "A"}, data_last_date="2026-08-15")
            second = append_experiment(path, {"rule": "A"}, data_last_date="2026-08-15")
            self.assertEqual(first["repeat_count"], 1)
            self.assertEqual(second["repeat_count"], 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("reproduction_fingerprint", saved)

    def test_ledger_converts_non_finite_result_to_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            append_experiment(
                path,
                {"rule": "A"},
                data_last_date="2026-08-15",
                result_summary={"median": float("nan"), "rows": [{"value": float("inf")}]},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["result_summary"]["median"])
            self.assertIsNone(saved["result_summary"]["rows"][0]["value"])

    def test_ledger_tolerates_malformed_previous_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            record = append_experiment(path, {"rule": "A"}, data_last_date="2026-08-15")
            self.assertEqual(record["malformed_previous_lines"], 1)


if __name__ == "__main__":
    unittest.main()
