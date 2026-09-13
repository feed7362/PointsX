"""Geometry snapshot gate: synthetic bodies must reproduce the committed expected output.

Fails on any change to calibration, width extraction, circumferences, validation
warnings or envelope corrections. If the change is intentional, regenerate:

    python tests/fixtures/synthetic_bodies.py --write-expected

and commit the updated JSON together with the code change, so the diff shows
exactly which measurements moved.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ABS_TOL = 1e-3  # values are rounded to 3-4 decimals; tolerate cross-platform float noise


def _load_generator():
    spec = importlib.util.spec_from_file_location("synthetic_bodies", FIXTURES / "synthetic_bodies.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _diff(expected, actual, path="") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in actual:
                out.append(f"{path}/{key}: missing (expected {expected[key]!r})")
            elif key not in expected:
                out.append(f"{path}/{key}: unexpected {actual[key]!r}")
            else:
                out.extend(_diff(expected[key], actual[key], f"{path}/{key}"))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != expected {len(expected)} ({actual!r})"]
        return [d for i, (e, a) in enumerate(zip(expected, actual)) for d in _diff(e, a, f"{path}[{i}]")]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
        return [] if math.isclose(expected, actual, abs_tol=ABS_TOL) else [f"{path}: {actual} != expected {expected}"]
    return [] if expected == actual else [f"{path}: {actual!r} != expected {expected!r}"]


class TestGeometrySnapshot(unittest.TestCase):
    def test_matches_expected(self):
        gen = _load_generator()
        expected = json.loads(gen.EXPECTED_PATH.read_text(encoding="utf-8"))
        diffs = _diff(expected, gen.snapshot())
        self.assertFalse(
            diffs,
            f"{len(diffs)} snapshot difference(s):\n  " + "\n  ".join(diffs[:40])
            + "\nIf intentional: python tests/fixtures/synthetic_bodies.py --write-expected",
        )

    def test_fixture_covers_core_circumferences(self):
        """Guard against a fixture that silently stops exercising the measurements we care about."""
        expected = json.loads(_load_generator().EXPECTED_PATH.read_text(encoding="utf-8"))
        for body_id, dump in expected.items():
            for mid in ("chest_circumference", "waist_circumference", "hip_circumference", "thigh_circumference"):
                self.assertIn(mid, dump["envelope"], f"{body_id} lost {mid}")


if __name__ == "__main__":
    unittest.main()
