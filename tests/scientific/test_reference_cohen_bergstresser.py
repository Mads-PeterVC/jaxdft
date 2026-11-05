"""Checks of the 1b reference data and the test-local NumPy model, independent of the
package. If these fail, the transcription, the tolerances or the theory post's
conventions are wrong, not the package."""

import pytest

from support import cohen_bergstresser as cbref
from support.reference import cb1966_si_table1
from support.tolerances import CB_TABLE_TOL_EV

TEST_ECUT = 6.0  # Ha, theory post 'Convergence with cutoff'


def test_reference_file_structure():
    table = cb1966_si_table1()
    assert table["lattice_constant_angstrom"] == 5.43
    assert table["units"] == "eV"
    ids = [row["id"] for row in table["splittings"]]
    assert len(ids) == len(set(ids)) == 7
    for row in table["splittings"]:
        assert row["tolerance_class"] in CB_TABLE_TOL_EV
        assert row["calculated"] is not None


@pytest.fixture(scope="module")
def numpy_model_splittings():
    return cbref.table1_splittings(TEST_ECUT)


@pytest.mark.parametrize("row", cb1966_si_table1()["splittings"], ids=lambda r: r["id"])
def test_numpy_model_reproduces_table1(numpy_model_splittings, row):
    """The conventions and tolerances in the theory post, applied in plain NumPy,
    reproduce the paper's calculated values."""
    assert abs(numpy_model_splittings[row["id"]] - row["calculated"]) <= CB_TABLE_TOL_EV[row["tolerance_class"]]
