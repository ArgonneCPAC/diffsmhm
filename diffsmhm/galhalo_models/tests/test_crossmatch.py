"""Test the crossmatch_integers function."""
import pytest
import numpy as np

from ..crossmatch import crossmatch_integers


fixed_seed = 43


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example1():
    """Test case where x has repeated entries with trivial overlap.

    The array x has unique entries.
    All y values are in x. All x values are in y.

    """
    x = np.array([1, 3, 5])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch_integers(x, y)

    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example2():
    """Test case where x has repeated entries with trivial overlap.

    The array x has repeated entries.
    All y values are in x. All x values are in y.

    """
    x = np.array([1, 3, 5, 3, 1, 1, 3, 5])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch_integers(x, y)

    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example3():
    """Test case where x has repeated entries with partial overlap.

    The array x has repeated entries.
    All y values are in x. Some x values are not in y.

    """
    x = np.array([0, 1, 3, 5, 3, -1, 1, 3, 5, -1])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch_integers(x, y)

    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example4():
    """Test nontrivial scenario of both entry repetition and non-trivial overlap.

    The array x has repeated entries.
    Some y has values are not in x. Some x values are not in y.
    This example comes from a hard-coded setup with a hard-coded answer.

    """
    x = np.array([1, 3, 5, 3, 1, -1, 3, 5, -10, -10])
    y = np.array([5, 1, 100, 20])
    x_idx, y_idx = crossmatch_integers(x, y)

    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example5():
    """Test nontrivial scenario of both entry repetition and non-trivial overlap.

    The array x has repeated entries.
    Some y has values are not in x. Some x values are not in y.
    This example comes from a randomized setup.

    """
    xmax = 100
    numx = 10000
    rng = np.random.RandomState(fixed_seed)
    x = rng.randint(0, xmax + 1, numx)

    y = np.arange(-xmax, xmax)[::10]
    rng.shuffle(y)

    x_idx, y_idx = crossmatch_integers(x, y)

    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_correctness_example6():
    """Test case where x and y have zero overlap."""
    x = np.array([-1, -5, -10])
    y = np.array([1, 2, 3, 4])
    x_idx, y_idx = crossmatch_integers(x, y)
    assert len(x_idx) == 0
    assert len(y_idx) == 0
    assert np.all(x[x_idx] == y[y_idx])


@pytest.mark.mpi_skip
def test_crossmatch_exception_handling1():
    """Verify that we raise the proper exception when y has repeated entries."""
    x = np.ones(5)
    y = np.ones(5)

    with pytest.raises(ValueError) as err:
        crossmatch_integers(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


@pytest.mark.mpi_skip
def test_crossmatch_exception_handling2():
    """Verify that we raise the proper exception when y has non-integer values."""
    x = np.ones(5)
    y = np.arange(0, 5, 0.5)

    with pytest.raises(ValueError) as err:
        crossmatch_integers(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


@pytest.mark.mpi_skip
def test_crossmatch_exception_handling3():
    """Verify that we raise the proper exception when y is multi-dimensional."""
    x = np.ones(5)
    y = np.arange(0, 6).reshape(2, 3)

    with pytest.raises(ValueError) as err:
        crossmatch_integers(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


@pytest.mark.mpi_skip
def test_crossmatch_exception_handling4():
    """Verify that we raise the proper exception when x has non-integer values."""
    x = np.arange(0, 5, 0.5)
    y = np.arange(0, 6)

    with pytest.raises(ValueError) as err:
        crossmatch_integers(x, y)
    substr = "Input array x must be a 1d sequence of integers"
    assert substr in err.value.args[0]


@pytest.mark.mpi_skip
def test_crossmatch_exception_handling5():
    """Verify that we raise the proper exception when x is multi-dimensional."""
    x = np.arange(0, 6).reshape(2, 3)
    y = np.arange(0, 6)

    with pytest.raises(ValueError) as err:
        crossmatch_integers(x, y)
    substr = "Input array x must be a 1d sequence of integers"
    assert substr in err.value.args[0]
