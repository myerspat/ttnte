"""The index bookkeeping of LayoutCross, checked through its read-only testing
aids on every test layout: core values decode in C order over the core's
modes, index sets are nested (each element a neighbor element merged with a
core value, the two fixing disjoint modes), block keys and block costs follow
their formulas, and pairs forced after a failed check are in every index set
of the passes after it."""

import itertools

import pytest
import torch

from .test_layout_cross import G, LAYOUTS, NG, NM, NS, _cross

# Column of each mode in comp, and its size
COLUMN = {"polar": 0, "azimuthal": 1, "space": 2, "energy": 3}
SIZE = [NM, NG, NS, G]


def _run(layout, **kw):
    c = _cross(layout, **kw)
    c.run()
    return c


def _rows(t):
    return [tuple(int(v) for v in row) for row in t.tolist()]


def _merge(a, b):
    """Union of two partial components, None if both fix a mode."""
    out = []
    for x, y in zip(a, b):
        if x >= 0 and y >= 0:
            return None
        out.append(x if x >= 0 else y)
    return tuple(out)


def _key(mu, ga, e):
    return ((mu + 1) * (NG + 1) + (ga + 1)) * (G + 1) + (e + 1)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_core_values_decode_in_c_order(layout):
    """Value j of a core is the C-order ravel of its modes (in the layout's order within
    the core), and leaves the other modes free."""
    c = _run(layout, tol=1e-4)
    for k, core in enumerate(layout.split("|")):
        modes = core.split(",")
        cols = [COLUMN[m] for m in modes]
        shape = [SIZE[col] for col in cols]
        comps = _rows(c.core_components(k))
        assert len(comps) == len(list(itertools.product(*map(range, shape))))
        for j, idx in enumerate(itertools.product(*map(range, shape))):
            expect = [-1] * 4
            for col, v in zip(cols, idx):
                expect[col] = v
            assert comps[j] == tuple(expect)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_index_sets_are_nested(layout):
    """left_[k+1] is a subset of left_[k] x core k and right_[k] of core k x
    right_[k+1], every element a merge of parts fixing disjoint modes; the end sets hold
    one element with every mode free."""
    c = _run(layout, tol=1e-4)
    K = len(c.cores)
    free = [(-1, -1, -1, -1)]
    assert _rows(c.index_set(True, 0)) == free
    assert _rows(c.index_set(False, K)) == free
    for k in range(K - 1):
        core = _rows(c.core_components(k))
        allowed = {_merge(a, b) for a in _rows(c.index_set(True, k)) for b in core}
        allowed.discard(None)
        for e in _rows(c.index_set(True, k + 1)):
            assert e in allowed, (k, e)
    for k in range(1, K):
        core = _rows(c.core_components(k))
        allowed = {_merge(a, b) for a in core for b in _rows(c.index_set(False, k + 1))}
        allowed.discard(None)
        for e in _rows(c.index_set(False, k)):
            assert e in allowed, (k, e)


def test_block_key_formula_and_cost():
    """block_key is the mixed-radix encoding of (mu + 1, gamma + 1, E + 1); after a run
    the store is released, so a block's cost is the number of fibers it covers (one per
    value of each free mode)."""
    c = _run("polar|azimuthal|space|energy", tol=1e-4)
    for mu, ga, e in itertools.product(
        [-1, 0, NM - 1], [-1, 3, NG - 1], [-1, 0, G - 1]
    ):
        key = c.block_key(mu, ga, e)
        assert key == _key(mu, ga, e)
        count = (NM if mu < 0 else 1) * (NG if ga < 0 else 1) * (G if e < 0 else 1)
        assert c.unswept_in_block(key) == count


@pytest.mark.parametrize("layout", LAYOUTS)
def test_forced_pairs_enter_every_index_set(layout):
    """Pairs forced by a failed check are, in the passes after it, in every left and
    right index set, projected to that set's modes."""
    # A tiny acceptance threshold makes every check fail, whatever the layout
    c = _run(layout, tol=1e-9, max_sweeps=3, stall_passes=0, holdout_factor=1e-3)
    K = len(c.cores)
    checks = [h["forced"] for h in c.history if h["phase"] == "backward"]
    # Pairs forced before the last check went through at least one more pass
    absorbed = c.forced_pairs[: checks[-2]] if len(checks) > 1 else []
    assert absorbed, "the checks should fail at this tolerance"
    modes_of = [[COLUMN[m] for m in core.split(",")] for core in layout.split("|")]

    def projected(p, cores):
        d, g = divmod(p, G)
        mu, ga = divmod(d, NG)
        held = {col for k in cores for col in modes_of[k]}
        return _key(
            mu if 0 in held else -1, ga if 1 in held else -1, g if 3 in held else -1
        )

    for first, last, left, k in [(0, k, True, k + 1) for k in range(K - 1)] + [
        (k, K - 1, False, k) for k in range(1, K)
    ]:
        keys = {_key(e[0], e[1], e[3]) for e in _rows(c.index_set(left, k))}
        expect = [projected(p, range(first, last + 1)) for p in c.forced_pairs]
        assert c.forced_block_keys(first, last) == expect
        for p in absorbed:
            assert projected(p, range(first, last + 1)) in keys, (left, k, p)


# =================================================================
# Candidate rows, block Grams and bases (recorded half passes)


def _recorded_run(layout, **kw):
    c = _cross(layout, **kw)
    c.set_recording(True)
    c.run()
    assert c.num_recorded == len(c.history)
    return c


def _failing(layout):
    """Every check fails: several passes, forced pairs and residual growth."""
    return _recorded_run(
        layout, tol=1e-9, max_sweeps=3, stall_passes=0, holdout_factor=1e-3
    )


def _core_value(c, layout):
    """Per core, the map from an element's components to the core's value j."""
    out = []
    for k, core in enumerate(layout.split("|")):
        cols = [COLUMN[m] for m in core.split(",")]
        comps = _rows(c.core_components(k))
        out.append(
            (
                cols,
                {tuple(comp[col] for col in cols): j for j, comp in enumerate(comps)},
            )
        )
    return out


def _j(value_of, m, e):
    cols, index = value_of[m]
    return index[tuple(e[col] for col in cols)]


def _last(c, forward):
    return max(i for i in range(c.num_recorded) if c.recorded_forward(i) == forward)


def _side(basis):
    return torch.ones(1, 1) if basis is None else basis


@pytest.mark.parametrize("layout", LAYOUTS)
def test_block_gram_paths_agree(layout):
    """Every Gram path (batched rows, per core group, factored H_A) gives the Gram of
    each block's candidate rows S_a K_j, formed here row by row."""

    c = _recorded_run(layout, tol=1e-4)
    K = len(c.cores)
    fwd, bwd = _last(c, True), _last(c, False)
    bonds = [(k, True) for k in range(K - 1)] + [(k, False) for k in range(1, K)]
    for k, left in bonds:
        rec = fwd if left else bwd
        core = c.recorded_cores(rec)[k]
        S = _side(c.recorded_bases(rec)[k if left else k + 1])
        side = _rows(c.index_set(left, k if left else k + 1))
        values = _rows(c.core_components(k))
        expect = {}
        for a, sa in enumerate(side):
            for j, cj in enumerate(values):
                t = S[a] @ core[:, j, :] if left else core[:, j, :] @ S[a]
                m = _merge(sa, cj)
                key = _key(m[0], m[1], m[3])
                expect[key] = expect.get(key, 0) + torch.outer(t, t)
        for factored in (False, True):
            grams, keys = c.candidate_grams(k, left, core, factored)
            assert sorted(keys) == sorted(expect), (k, left)
            for g, key in zip(grams, keys):
                ref = expect[key]
                assert torch.allclose(g, ref, rtol=1e-12, atol=1e-12 * ref.norm()), (
                    k,
                    left,
                    factored,
                )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bases_are_the_cores_at_the_index_sets(layout):
    """After every half pass, the basis rows chosen with the index sets are the train's
    partial products there: B_k[e] = G_1(e) ...

    G_{k-1}(e)
    forward, C_k[e] = G_k(e) ... G_K(e) backward (nestedness).
    """
    c = _failing(layout)
    K = len(c.cores)
    value_of = _core_value(c, layout)
    for i in range(c.num_recorded):
        forward = c.recorded_forward(i)
        cores, bases = c.recorded_cores(i), c.recorded_bases(i)
        for k in range(1, K):
            if bases[k] is None:
                continue
            elems = _rows(c.recorded_index_set(i, forward, k))
            assert bases[k].shape[0] == len(elems)
            for row, e in enumerate(elems):
                v = torch.ones(1, 1)
                if forward:
                    for m in range(k):
                        v = v @ cores[m][:, _j(value_of, m, e), :]
                    got = v[0]
                else:
                    for m in range(K - 1, k - 1, -1):
                        v = cores[m][:, _j(value_of, m, e), :] @ v
                    got = v[:, 0]
                ref = bases[k][row]
                assert torch.allclose(got, ref, atol=1e-10 * max(ref.norm(), 1.0)), (
                    i,
                    k,
                    e,
                )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_index_sets_only_grow_within_a_run(layout):
    """Every element of every index set stays in it in the half passes after, and the
    residual-grown blocks are in the last core's right set."""
    c = _failing(layout)
    K = len(c.cores)
    assert c.num_recorded >= 4
    for i in range(c.num_recorded - 1):
        for left in (True, False):
            for k in range(1, K):
                before = set(_rows(c.recorded_index_set(i, left, k)))
                after = set(_rows(c.recorded_index_set(i + 1, left, k)))
                assert before <= after, (i, left, k, before - after)
    last = {_key(e[0], e[1], e[3]) for e in _rows(c.index_set(False, K - 1))}
    assert set(c.grown_block_keys(K - 1, K - 1)) <= last


def test_residual_growth_happens():
    """The failing runs above do grow the last bond (so the check is not vacuous)."""
    c = _failing("polar|azimuthal|space|energy")
    assert c.grown_block_keys(3, 3)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_factored_grams_select_as_well(layout):
    """Selection on all-factored Grams reaches the same accuracy."""
    from .test_layout_cross import REF, _rel

    c = _cross(layout)
    c.set_factored_grams(True)
    c.run()
    assert c.converged
    full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
    assert _rel(full, REF) < 20 * 1e-6
