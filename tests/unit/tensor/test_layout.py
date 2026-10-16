import pytest

from ttnte.tensor import Dimension, Index, Layout


def test_known_indices_share_labels():
    assert Index.polar(4) == Index.from_name("mu", 4)
    assert Index.polar(4) != Index.polar(8)
    assert Index.polar(4).same_label(Index.polar(8))
    assert Index.from_name("E").dimension == Dimension.ENERGY
    assert Index.from_name("s").dimension == Dimension.SPACE
    assert Index(5, "custom").dimension == Dimension.OTHER
    with pytest.raises(RuntimeError):
        Index.from_name("theta")


def test_parse_string_and_aliases():
    a = Layout("polar | azimuthal | x,y | energy")
    b = Layout("mu|gamma|x,y|E")
    assert a == b
    assert str(a) == "polar|azimuthal|x,y|energy"
    assert a.num_cores == 4 and a.core_of(Dimension.Y) == 2


def test_index_lists_equal_strings():
    by_index = Layout(
        [[Index.polar(), Index.azimuthal()], [Index.space()], [Index.energy()]]
    )
    assert by_index == Layout("polar,azimuthal|space|energy")


def test_canonical_orders():
    assert str(Layout.canonical(1)) == "polar|x|energy"
    assert str(Layout.canonical(2)) == "polar|azimuthal|x|y|energy"
    assert str(Layout.canonical(3, False)) == "polar|azimuthal|space|energy"


def test_any_order_validates():
    for spec, ndim in [
        ("energy|space|azimuthal,polar", 2),
        ("x|y|z|energy|azimuthal|polar", 3),
        ("polar,x,energy", 1),
    ]:
        Layout(spec).validate(ndim)


@pytest.mark.parametrize(
    "spec, ndim",
    [
        ("polar|x|energy", 2),  # missing azimuthal and y
        ("polar|azimuthal|x|energy", 1),  # azimuthal in 1-D
        ("polar|azimuthal|x|y|space|energy", 2),  # both space forms
        ("polar|x|energy|y|azimuthal", 2),  # space not contiguous
        ("polar|azimuthal|x|y", 2),  # no energy
    ],
)
def test_invalid_layouts(spec, ndim):
    with pytest.raises(RuntimeError):
        Layout(spec).validate(ndim)


def test_malformed_strings():
    for spec in ["polar||energy", "polar,|energy", "polar|polar", "polar|theta"]:
        with pytest.raises(RuntimeError):
            Layout(spec)


def test_merge_space_and_sizes():
    lay = Layout("polar,x|y|energy").with_sizes(
        {Dimension.POLAR: 4, Dimension.X: 3, Dimension.Y: 5, Dimension.ENERGY: 7}
    )
    assert lay.core_size(0) == 12
    merged = lay.merge_space()
    assert str(merged) == "polar,space|energy"
    assert merged.cores[0][1].size == 15
    assert str(Layout("polar|x,y|energy").merge_space()) == "polar|space|energy"
