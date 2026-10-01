"""The command line entry point runs and prints numbers matching the library."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def invoke(*args):
    """Run run.py as a subprocess and return (returncode, stdout + stderr)."""
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), *args],
                           capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_cournot_subcommand_prints_the_nash_quantity():
    code, out = invoke("cournot", "-a", "100", "-b", "1", "-c", "20", "-n", "3")
    assert code == 0, out
    assert "20.0000" in out  # (100-20)/4 per firm
    assert "60.0000" in out  # total Q


def test_cournot_subcommand_includes_stackelberg_for_n_equals_2():
    code, out = invoke("cournot", "-a", "100", "-b", "1", "-c", "20", "-n", "2")
    assert code == 0, out
    assert "Stackelberg" in out


def test_nash_subcommand_finds_the_mixed_equilibrium():
    code, out = invoke("nash", "--game", "matching-pennies")
    assert code == 0, out
    assert "0.5" in out


def test_bayes_subcommand_matches_the_library():
    code, out = invoke("bayes", "-a", "100", "--c1", "20", "--cH", "32", "--cL", "20", "--theta", "0.5")
    assert code == 0, out
    assert "28.6667" in out


def test_repeated_subcommand_prints_the_duopoly_threshold():
    code, out = invoke("repeated", "--cournot", "-n", "2", "-a", "100", "-b", "1", "-c", "20")
    assert code == 0, out
    assert "0.5294" in out


def test_missing_subcommand_fails():
    code, out = invoke()
    assert code != 0
