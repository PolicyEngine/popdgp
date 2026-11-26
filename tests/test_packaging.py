"""What a consumer of the published package relies on.

The version string agrees with the installed metadata, and the quick start in
the README is the script in ``examples/`` and runs.
"""

from __future__ import annotations

import runpy
from importlib import metadata
from pathlib import Path

import popdgp

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.py"


def test_version_matches_the_installed_metadata() -> None:
    """``pyproject.toml`` and ``popdgp.__version__`` name the same version."""
    assert popdgp.__version__ == metadata.version("popdgp")


def test_readme_quick_start_is_the_example_script() -> None:
    """The README shows the example verbatim, so the two cannot drift apart."""
    readme = (ROOT / "README.md").read_text()
    assert f"```python\n{QUICKSTART.read_text()}```" in readme


def test_quick_start_runs_and_reports_every_block(capsys) -> None:
    """The example scores a candidate and its floor on all four blocks."""
    runpy.run_path(str(QUICKSTART), run_name="__main__")
    printed = capsys.readouterr().out
    for metric in (
        "energy_distance",
        "c2st_auc",
        "prdc_coverage",
        "w1_over_sd.wealth",
        "q99_ratio.wealth",
    ):
        assert metric in printed
    assert "value_candidate" in printed and "value_floor" in printed
