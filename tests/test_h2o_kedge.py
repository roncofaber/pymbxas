import logging

import numpy as np
import pytest
import ase.build
from ase import units
from pymbxas.calculators.pyscf import PySCFMBXAS
from pymbxas.config import (
    CalculationConfig, CheckpointConfig, LoggingConfig, RuntimeConfig,
)
from pymbxas.build.structure import ase_to_mole
from pymbxas.mbxas.maxvol import sherman_morrison_row_update


def test_pyscf_builder_forwards_scf_controls():
    from pyscf import gto
    from pymbxas.build.input_pyscf import make_pyscf_calculator
    from pymbxas.spectra import _molecule_settings

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = make_pyscf_calculator(
        mol, xc="lda", calc_type="RKS", max_cycle=17,
        conv_tol=1e-6, grid_level=1)
    assert calc.max_cycle == 17
    assert calc.conv_tol == pytest.approx(1e-6)
    assert calc.grids.level == 1
    assert _molecule_settings({
        "charge": 0, "spin": 0, "basis": "sto-3g", "pbc": False,
        "xc": "lda", "calc_type": "UKS", "scf_max_cycle": 150,
    }) == {"charge": 0, "spin": 0, "basis": "sto-3g", "pbc": False}


def test_non_aufbau_core_hole_and_xch_target_selection():
    from pymbxas.calculators.excitation import _xch_target_index
    from pymbxas.io.data import pyscf_data
    from pymbxas.mbxas.mbxas import core_hole_index, occ_unocc_indices

    # The core hole is MO 3, while the first ordinary virtual is MO 1.  This
    # reproduces the non-Aufbau ordering that exposed the 6fda XCH bug.
    mb_overlap = np.zeros((6, 6))
    mb_overlap[3, 2] = 0.99
    mb_overlap[0, 2] = 0.02
    fch_occ = np.array([1, 0, 1, 0, 1, 0])
    gs_occ = np.array([1, 1, 1, 1, 0, 0])
    hole = core_hole_index(mb_overlap, fch_occ, core_orb_idx=2)
    assert hole == 3

    occ_gs, occ_fch, virtual = occ_unocc_indices(
        gs_occ, fch_occ, core_orb_idx=2, core_hole_idx=hole)
    assert np.array_equal(occ_gs, [0, 1, 3])
    assert np.array_equal(occ_fch, [0, 2, 4])
    assert np.array_equal(virtual, [1, 5])

    data = pyscf_data.from_arrays(
        mol=None, e_tot=0.0, nelec=(3, 3),
        mo_coeff=np.stack([np.eye(6), np.eye(6)]),
        mo_occ=np.stack([fch_occ, fch_occ]),
        mo_energy=np.stack([
            np.array([-3.0, 0.2, -1.0, -20.0, -0.5, 0.7]),
            np.array([-3.0, 0.2, -1.0, -20.0, -0.5, 0.7]),
        ]))
    assert _xch_target_index(data, core_hole_idx=hole, channel=1) == 1


def test_sherman_morrison_row_update():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.normal(size=(n, n))
    A_inv = np.linalg.inv(A)

    for row_idx in range(n):
        new_row = rng.normal(size=n)
        A_new, A_inv_new = sherman_morrison_row_update(A, A_inv, row_idx, new_row)

        A_expected = A.copy()
        A_expected[row_idx] = new_row
        assert np.allclose(A_new, A_expected), \
            f"row {row_idx}: updated matrix does not match the row replacement"

        A_inv_expected = np.linalg.inv(A_expected)
        assert np.allclose(A_inv_new, A_inv_expected, atol=1e-10), \
            f"row {row_idx}: Sherman-Morrison inverse disagrees with np.linalg.inv from scratch"

    # A near-singular update (new row duplicates another row) must raise,
    # not silently return garbage.
    A_dup = A.copy()
    with pytest.raises(np.linalg.LinAlgError):
        sherman_morrison_row_update(A, A_inv, 0, A_dup[1])


def test_exact_overlap_doubles_match_minor_identity():
    from pymbxas.mbxas.shakeup import overlap_sticks

    rng = np.random.default_rng(1)
    n_occ, n_unocc = 4, 4
    AMat = np.eye(n_occ) + 0.01 * rng.normal(size=(n_occ, n_occ))
    APrimeMat = 0.01 * rng.normal(size=(n_unocc, n_occ))
    # Make one specific 2-swap configuration (valence {0,2} -> conduction
    # {1,3}) dominant so the search is guaranteed to find it.
    APrimeMat[1, 0] = 0.9
    APrimeMat[3, 2] = 0.9

    K = APrimeMat @ np.linalg.inv(AMat)
    eps_occ = np.array([-1.0, -1.2, -1.4, -1.6])
    eps_unocc = np.array([0.5, 0.6, 0.7, 0.8])

    sticks = overlap_sticks(AMat, APrimeMat, eps_occ, eps_unocc, order=2)
    delta_e, weight = sticks.energy, sticks.weight
    assert len(weight) == 36

    expected_weight = (np.abs(np.linalg.det(K[np.ix_([1, 3], [0, 2])])) ** 2
                       * np.abs(np.linalg.det(AMat)) ** 2)
    expected_delta_e = (eps_unocc[1] + eps_unocc[3]) - (eps_occ[0] + eps_occ[2])
    assert any(abs(w - expected_weight) < 1e-10 for w in weight), \
        "no discovered order-2 config matches the hand-computed dominant 2x2 minor"
    idx = np.argmin(np.abs(weight - expected_weight))
    assert abs(delta_e[idx] - expected_delta_e) < 1e-10, \
        "the matching config's delta_e does not match the expected energy sum"


def test_h2o_oxygen_kedge(tmp_path, monkeypatch):
    structure = ase.build.molecule("H2O")

    obj = PySCFMBXAS(
        structure,
        calculation=CalculationConfig(xc="lda", basis="def2-svpd"),
        checkpoint=None,
    )

    obj.run(
        "O", runtime=RuntimeConfig(work_directory=tmp_path),
        logging=LoggingConfig(
            pymbxas_verbosity=1, pyscf_verbosity=0,
            pyscf_logfile="pyscf.log", pyscf_console=False))

    exc = obj.excitations[0]
    assert "BEGIN PyMBXAS SCF" in obj.output
    assert "stage      : GS" in obj.output
    assert "stage      : FCH" in exc.output["fch"]
    assert "site       : O:0" in exc.output["fch"]
    assert "channel    : beta" in exc.output["fch"]
    assert "stage      : XCH" in exc.output["xch"]
    raw_log = (tmp_path / "pyscf.log").read_text(encoding="utf-8")
    assert raw_log.count("BEGIN PyMBXAS SCF") == 3
    assert raw_log.count("stage      : GS") == 1
    assert raw_log.count("stage      : FCH") == 1
    assert raw_log.count("stage      : XCH") == 1
    gs = obj.gs_data
    fch = exc.data["fch"]
    xch = exc.data["xch"]
    ch = exc.channel
    S = gs.mol.intor("int1e_ovlp")
    Ha = units.Ha

    assert len(obj.excitations) == 1, "Expected exactly one excitation"
    assert exc.ato_idx == 0, "Expected excited atom index 0 (oxygen)"
    assert exc.channel == 1, "Expected excited channel 1 (beta)"

    assert gs.nelec == (5, 5), f"Expected GS nelec (5, 5), got {gs.nelec}"
    assert fch.nelec == (5, 4), f"Expected FCH nelec (5, 4), got {fch.nelec}"
    assert xch.nelec == (5, 5), f"Expected XCH nelec (5, 5), got {xch.nelec}"
    assert tuple(np.count_nonzero(xch.mo_occ == 1, axis=1)) == xch.nelec, \
        "Converged XCH occupations do not match the molecule electron counts"

    hole = np.where(fch.mo_occ[ch] == 0)[0][0]
    assert hole == 0, f"Expected hole at index 0, got {hole}"

    c_hole = fch.mo_coeff[ch][:, hole]
    mulliken_weight = np.abs(c_hole * (S @ c_hole))

    ao_labels = np.array(gs.mol.ao_labels(fmt=False), dtype=object)
    o1s_mask = np.array([("O" in str(label[1]) and "1s" in str(label[2])) for label in ao_labels])
    o1s_weight = mulliken_weight[o1s_mask].sum() / mulliken_weight.sum() if o1s_mask.any() else 0.0

    assert o1s_weight > 0.95, f"Expected O 1s weight > 0.95, got {o1s_weight:.4f}"

    c_excited = fch.mo_coeff[ch][:, hole]
    overlap_variational = abs(c_excited @ S @ gs.mo_coeff[ch][:, exc.orb_idx])
    assert overlap_variational > 0.99, f"Variational collapse detected: overlap {overlap_variational:.5f} < 0.99"

    occ_gs = np.setdiff1d(np.where(gs.mo_occ[ch] == 1)[0], [exc.orb_idx])
    occ_fch = np.where(fch.mo_occ[ch] == 1)[0]
    A = (fch.mo_coeff[ch].T @ S @ gs.mo_coeff[ch])[np.ix_(occ_fch, occ_gs)]

    assert A.shape[0] == A.shape[1], f"Matrix A not square: {A.shape}"

    det_A = np.linalg.det(A)
    # Orbital sign is gauge-arbitrary (CPU and GPU eigensolvers can pick
    # opposite signs for the same MO), so only |det(A)| is a real invariant.
    assert abs(abs(det_A) - 0.9486) < 0.05, f"|det(A)| = {abs(det_A):.4f}, expected ~0.9486"

    cond_A = np.linalg.cond(A)
    assert cond_A < 2, f"Matrix A ill-conditioned: cond(A) = {cond_A:.3f} > 2"

    uno_fch = np.where(fch.mo_occ[ch] == 0)[0][1:]
    Ap = (fch.mo_coeff[ch].T @ S @ gs.mo_coeff[ch])[np.ix_(uno_fch, occ_gs)]
    K = Ap @ np.linalg.inv(A)
    r = gs.mol.intor("int1e_r")
    x = np.einsum('xmn,m,nf->xf', r, c_hole, fch.mo_coeff[ch])
    amp_recomputed = det_A * (x[:, uno_fch] - (K @ x[:, occ_fch].T).T)

    amp_library = exc.mbxas["absorption"]
    # Independent restatement of the determinant formula from dev/method.md; do not simplify by calling the library function.
    assert amp_recomputed.shape == amp_library.shape, f"Amplitude shape mismatch: {amp_recomputed.shape} vs {amp_library.shape}"

    max_diff = np.max(np.abs(amp_recomputed - amp_library))
    assert max_diff < 1e-12, f"Amplitude disagreement: max diff {max_diff:.2e}, expected < 1e-12"

    xch_energy_shift = (xch.e_tot - gs.e_tot) * Ha
    min_transition = exc.mbxas["energies"].min() * Ha
    assert abs(min_transition - xch_energy_shift) < 1e-8, f"XCH alignment mismatch: {min_transition:.6f} vs {xch_energy_shift:.6f} eV"

    first_transition_ev = exc.mbxas["energies"][0] * Ha
    assert 524.14 < first_transition_ev < 534.14, f"First transition at {first_transition_ev:.2f} eV, expected ~529.14 eV (within 5 eV)"

    st2 = structure.copy()
    st2.translate([5.0, 3.0, -2.0])
    mol2 = ase_to_mole(st2, 0, 0, basis="def2-svpd", pbc=False, verbose=0, print_output=False)
    r2 = mol2.intor("int1e_r")
    x2 = np.einsum('xmn,m,nf->xf', r2, c_hole, fch.mo_coeff[ch])

    virtual_block_orig = x[:, uno_fch]
    virtual_block_trans = x2[:, uno_fch]
    virtual_diff = np.max(np.abs(virtual_block_orig - virtual_block_trans))
    # Origin independence holds because both orbitals come from the same FCH calculation.
    assert virtual_diff < 1e-10, f"Virtual dipoles changed under translation: {virtual_diff:.2e}, expected < 1e-10"

    # Mole.build kwargs are validated against the installed PySCF API:
    # magmom is valid and retained, while a typo fails immediately.
    mol_mag = ase_to_mole(
        st2, 0, 0, basis="def2-svpd", pbc=False, verbose=0,
        print_output=False, magmom=[0, 0, 0])
    assert np.array_equal(mol_mag.magmom, [0, 0, 0])
    with pytest.raises(TypeError, match="not supported.*Mole.build"):
        ase_to_mole(st2, 0, 0, basis="def2-svpd", pbc=False, verbose=0,
                    print_output=False, this_kwarg_does_not_exist=True)

    E, I = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5)

    assert len(E) == len(I), f"Energy and intensity arrays have different lengths: {len(E)} vs {len(I)}"
    assert np.all(np.isfinite(I)), "Intensity contains non-finite values"
    assert np.min(I) >= 0, f"Negative intensity found: min = {np.min(I)}"
    assert np.max(I) > 0, "All intensities are zero"

    max_idx = np.argmax(I)
    assert E[0] <= E[max_idx] <= E[-1], f"Spectrum maximum at {E[max_idx]:.1f} eV outside erange [520, 560]"

    # Intensity carries the photon-energy prefactor sigma(omega) ~ omega * |M|^2
    # (Eq. 4, PRB 107, 035146). Independent restatement; do not simplify by
    # calling amp2int().
    per_transition_intensity = exc.mbxas["energies"] * np.mean(amp_library**2, axis=0)
    spectra_direct = obj.to_spectra(index=0)
    assert np.allclose(spectra_direct.amp2int(), per_transition_intensity, atol=1e-15), \
        "Spectra.amp2int() does not include the omega prefactor"

    # PySCFMBXAS.get_mbxas_spectra and Spectra.get_mbxas_spectra must stay
    # numerically identical (see dev/method.md gotcha on the three
    # get_mbxas_spectra implementations).
    E_spectra, I_spectra = spectra_direct.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.allclose(E, E_spectra) and np.allclose(I, I_spectra, atol=1e-12), \
        "PySCFMBXAS.get_mbxas_spectra and Spectra.get_mbxas_spectra disagree"

    assert amp_library.shape[0] == 3, f"Amplitude first dimension should be 3 (Cartesian), got {amp_library.shape[0]}"
    assert amp_library.shape[1] == len(exc.mbxas["energies"]), f"Amplitude transitions mismatch: {amp_library.shape[1]} vs {len(exc.mbxas['energies'])}"

    spectra_fields = obj.to_spectra(index=0)
    assert np.array_equal(spectra_fields._mb_overlap, exc.mbxas["mb_overlap"]), \
        "Spectra._mb_overlap does not match the excitation's mb_overlap"
    assert np.array_equal(spectra_fields._fch_mo_energy, fch.mo_energy), \
        "Spectra._fch_mo_energy does not match the FCH mo_energy"
    assert np.array_equal(spectra_fields._gs_mo_occ, gs.mo_occ), \
        "Spectra._gs_mo_occ does not match the GS mo_occ"
    assert spectra_fields._core_orb_idx == exc.orb_idx, \
        f"Spectra._core_orb_idx {spectra_fields._core_orb_idx} != exc.orb_idx {exc.orb_idx}"

    from pymbxas.mbxas.shakeup import shakeup_sticks, shakeup_spectrum
    from pymbxas.mbxas.mbxas import build_A_K

    occ_idxs_gs_ch = np.setdiff1d(np.where(gs.mo_occ[ch] == 1)[0], [exc.orb_idx])
    occ_idxs_fch_ch = np.where(fch.mo_occ[ch] == 1)[0]
    uno_idxs_fch_ch = np.where(fch.mo_occ[ch] == 0)[0][1:]
    mb_overlap_ch = exc.mbxas["mb_overlap"][ch]
    AMat_ch, _, K_ch, APrimeMat_ch = build_A_K(mb_overlap_ch, occ_idxs_fch_ch, occ_idxs_gs_ch, uno_idxs_fch_ch)
    eps_occ_ch = fch.mo_energy[ch][occ_idxs_fch_ch]
    eps_unocc_ch = fch.mo_energy[ch][uno_idxs_fch_ch]

    # Order-1 overlap sticks include the reference determinant weight.
    e1, w1 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert e1.shape == w1.shape == (len(occ_idxs_fch_ch) * len(uno_idxs_fch_ch),), \
        f"order=1 shake-up stick count mismatch: {e1.shape} vs expected {(len(occ_idxs_fch_ch)*len(uno_idxs_fch_ch),)}"
    w1_manual = np.abs(np.linalg.det(AMat_ch)) ** 2 * np.abs(K_ch) ** 2
    assert np.allclose(np.sort(w1), np.sort(w1_manual.ravel()), atol=1e-14), \
        "order=1 shake-up weights do not match |det(A)|^2 |K_vc|^2"

    # For order 1 the any-negative-constituent rule is simply delta_e < 0.
    e1_down, w1_down = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.all(e1_down < 0), "shakedown_only=True should keep only negative delta_e sticks"
    manual_mask = e1 < 0
    assert np.array_equal(np.sort(e1_down), np.sort(e1[manual_mask])), \
        "shakedown_only=True should match a manual delta_e<0 filter of the unfiltered order-1 sticks"
    assert np.array_equal(np.sort(w1_down), np.sort(w1[manual_mask])), \
        "shakedown_only=True should keep the matching weights unchanged"

    # Order 2 exhaustively enumerates all antisymmetrized 2x2 minors of K.
    e2, w2 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert w2.shape == e2.shape

    # Higher requested overlap orders are also exact (and combinatorial).
    e3, w3 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=3)
    assert e3.shape == w3.shape

    # shakeup_spectrum: explicit order=1 includes only order 1; explicit
    # order=2 always includes both orders (no silent auto-downgrade)
    de1, dw1, orders1 = shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert orders1 == [1], f"explicit order=1 should include only order 1, got {orders1}"
    de2, dw2, orders2 = shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders2 == [1, 2], f"explicit order=2 should include orders [1, 2], got {orders2}"
    assert len(de2) == len(e1) + len(e2), "order=2 spectrum should concatenate order-1 and order-2 sticks"

    # The incomplete heuristic auto mode was removed.
    if w2.sum() > 0:
        assert w2.sum() < w1.sum(), \
            f"order-2 total shake-up probability ({w2.sum():.3e}) should be smaller than order-1 ({w1.sum():.3e})"
    with pytest.raises(ValueError):
        shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order="auto", tol=0.01)

    from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

    sticks_by_order_2, orders_by_order_2 = shakeup_sticks_by_order(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders_by_order_2 == [1, 2], f"expected orders [1, 2], got {orders_by_order_2}"
    assert np.array_equal(sticks_by_order_2[1][0], e1) and np.array_equal(sticks_by_order_2[1][1], w1), \
        "shakeup_sticks_by_order order-1 entry should match shakeup_sticks(order=1)"
    assert np.array_equal(sticks_by_order_2[2][0], e2) and np.array_equal(sticks_by_order_2[2][1], w2), \
        "shakeup_sticks_by_order order-2 entry should match shakeup_sticks(order=2)"

    de2_from_dict = np.concatenate([sticks_by_order_2[k][0] for k in orders_by_order_2])
    dw2_from_dict = np.concatenate([sticks_by_order_2[k][1] for k in orders_by_order_2])
    assert np.array_equal(de2_from_dict, de2) and np.array_equal(dw2_from_dict, dw2), \
        "shakeup_spectrum(order=2) must equal the concatenation of shakeup_sticks_by_order's entries"

    sticks_by_order_down, _ = shakeup_sticks_by_order(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.array_equal(np.sort(sticks_by_order_down[1][0]), np.sort(e1_down)), \
        "shakeup_sticks_by_order should forward shakedown_only to shakeup_sticks"

    from pymbxas.mbxas.shakeup import combine_cross_channel_sticks

    sticks_a = {1: (np.array([1.0, 2.0]), np.array([0.1, 0.2]))}
    sticks_b = {1: (np.array([3.0]), np.array([0.4]))}
    de_cross, dw_cross = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order=2)
    expected_e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_w = np.array([0.1, 0.2, 0.4, 0.04, 0.08])
    assert np.array_equal(np.sort(de_cross), np.sort(expected_e)), \
        f"combine_cross_channel_sticks energies mismatch: {sorted(de_cross)} vs {sorted(expected_e)}"
    assert np.allclose(np.sort(dw_cross), np.sort(expected_w), atol=1e-15), \
        f"combine_cross_channel_sticks weights mismatch: {sorted(dw_cross)} vs {sorted(expected_w)}"

    de_cap1, dw_cap1 = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order=1)
    assert len(de_cap1) == 3, \
        f"max_total_order=1 should keep 3 sticks (2 pure-a + 1 pure-b, dropping the (1,1) cross term), got {len(de_cap1)}"

    de_solo, dw_solo = combine_cross_channel_sticks(sticks_a, {}, max_total_order=1)
    assert np.array_equal(np.sort(de_solo), np.sort(sticks_a[1][0])) \
        and np.array_equal(np.sort(dw_solo), np.sort(sticks_a[1][1])), \
        "combine_cross_channel_sticks with an empty spectator dict should reduce to the excited channel's own sticks"

    de_empty, dw_empty = combine_cross_channel_sticks({}, {}, max_total_order=0)
    assert len(de_empty) == 0 and len(dw_empty) == 0, \
        "combine_cross_channel_sticks with both dicts empty should return empty arrays"

    from pymbxas.mbxas.shakeup import broaden_shakeup, convolve_shakeup

    # broaden_shakeup with empty sticks reduces to a single normalized
    # Gaussian at delta_e=0 (the implicit n=0 "no extra shake-up" term)
    egrid_probe = np.linspace(-5, 5, 2001)
    kernel_empty = broaden_shakeup(np.empty(0), np.empty(0), egrid_probe, sigma=0.5)
    de_probe = egrid_probe[1] - egrid_probe[0]
    assert abs(kernel_empty.sum() * de_probe - 1.0) < 1e-6, \
        f"empty-sticks shake-up kernel should integrate to 1, got {kernel_empty.sum()*de_probe:.6f}"
    assert egrid_probe[np.argmax(kernel_empty)] == pytest.approx(0.0, abs=de_probe), \
        "empty-sticks shake-up kernel should peak at delta_e=0"

    # convolve_shakeup with empty sticks must leave the main spectrum
    # unchanged (the no-extra-pair limit)
    main_probe = np.exp(-0.5 * (egrid_probe / 1.0) ** 2)
    convolved_empty = convolve_shakeup(egrid_probe, main_probe, np.empty(0), np.empty(0), sigma=0.5)
    assert np.allclose(convolved_empty, main_probe, atol=1e-3), \
        "convolving with an empty shake-up spectrum should not change the main spectrum"

    # a single shake-up stick at a known offset should shift probability
    # mass to that offset. The kernel is not unit-normalized: equal reference
    # and satellite weights double the integrated intensity.
    stick_de = np.array([2.0])
    stick_w = np.array([1.0])  # equal weight to the n=0 term, for an easy 50/50 check
    convolved_one = convolve_shakeup(egrid_probe, main_probe, stick_de, stick_w, sigma=0.5)
    assert np.trapezoid(convolved_one, egrid_probe) == pytest.approx(
        2 * np.trapezoid(main_probe, egrid_probe), rel=0.05), \
        "unnormalized equal-weight reference and satellite should double the area"
    # half the conserved intensity should now sit near delta_e=+2 rather
    # than at the original peak (equal-weight n=0 vs n=1 split)
    mass_near_peak = np.trapezoid(convolved_one[(egrid_probe > -1) & (egrid_probe < 1)], egrid_probe[(egrid_probe > -1) & (egrid_probe < 1)])
    mass_near_satellite = np.trapezoid(convolved_one[(egrid_probe > 1) & (egrid_probe < 3)], egrid_probe[(egrid_probe > 1) & (egrid_probe < 3)])
    assert mass_near_satellite > 0.3 * mass_near_peak, \
        "equal-weight single shake-up stick should move a comparable amount of intensity to the satellite"

    from pymbxas.mbxas.mbxas import occ_unocc_indices

    # occ_unocc_indices must reproduce the same three index arrays already
    # computed by hand at the top of this test
    occ_gs_h, occ_fch_h, uno_fch_h = occ_unocc_indices(gs.mo_occ[ch], fch.mo_occ[ch], exc.orb_idx)
    assert np.array_equal(occ_gs_h, occ_gs), "occ_unocc_indices GS occupied indices mismatch"
    assert np.array_equal(occ_fch_h, occ_fch), "occ_unocc_indices FCH occupied indices mismatch"
    assert np.array_equal(uno_fch_h, uno_fch), "occ_unocc_indices FCH unoccupied indices mismatch"

    from pymbxas.mbxas.mbxas import spectator_occ_unocc_indices

    spec_ch = 1 - ch
    occ_gs_spec_h, occ_fch_spec_h, uno_fch_spec_h = spectator_occ_unocc_indices(
        gs.mo_occ[spec_ch], fch.mo_occ[spec_ch])
    assert np.array_equal(occ_gs_spec_h, np.where(gs.mo_occ[spec_ch] == 1)[0]), \
        "spectator_occ_unocc_indices GS occupied indices mismatch (no core orbital should be removed)"
    assert np.array_equal(occ_fch_spec_h, np.where(fch.mo_occ[spec_ch] == 1)[0]), \
        "spectator_occ_unocc_indices FCH occupied indices mismatch"
    assert np.array_equal(uno_fch_spec_h, np.where(fch.mo_occ[spec_ch] == 0)[0]), \
        "spectator_occ_unocc_indices FCH unoccupied indices mismatch (no core-hole index should be dropped)"
    assert len(occ_gs_spec_h) == len(occ_fch_spec_h), \
        "spectator channel electron count should be unchanged between GS and FCH"

    with pytest.raises(ValueError):
        spectator_occ_unocc_indices(gs.mo_occ[ch], fch.mo_occ[ch])  # excited channel has a core hole

    # verbosity level 5 must configure the pymbxas logger to a strictly
    # more detailed level than 4 (previously both mapped to logging.DEBUG,
    # making them indistinguishable)
    from pymbxas.io.config import configure_logger, TRACE
    assert TRACE < logging.DEBUG, f"TRACE ({TRACE}) should be below DEBUG ({logging.DEBUG})"
    configure_logger(5)
    assert logging.getLogger("pymbxas").level == TRACE, \
        "verbosity level 5 should configure the pymbxas logger to TRACE, not DEBUG"
    configure_logger(3)  # restore a normal level so later output isn't flooded

    h5_path = obj.save(tmp_path / "roundtrip.h5")

    from pyscf.scf import chkfile as pyscf_chkfile
    from pymbxas.calculators.pyscf import PySCFMBXAS as _PySCFMBXAS
    from pymbxas.mbxas.mbxas import run_MBXAS_pyscf
    from pymbxas.spectra import Spectra

    mol_chk, scf_chk = pyscf_chkfile.load_scf(h5_path)
    assert mol_chk.natm == 3, "Checkpoint is not readable as a PySCF chkfile"
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(gs.mo_coeff)), \
        "chkfile-read GS coefficients differ from the in-memory ones"

    back = _PySCFMBXAS.load(h5_path)
    assert back._ran_GS is True, "Reloaded object does not report a finished ground state"
    assert back.excited_idxs == [0], f"Expected excited atom [0], got {back.excited_idxs}"

    b_gs  = back.gs_data
    b_exc = back.excitations[0]
    b_fch = b_exc.data["fch"]
    b_xch = b_exc.data["xch"]

    assert np.array_equal(b_gs.mo_coeff, gs.mo_coeff), "GS coefficients changed across a save/load"
    assert np.array_equal(b_gs.mo_occ, gs.mo_occ), "GS occupations changed across a save/load"
    assert b_gs.e_tot == gs.e_tot, "GS energy changed across a save/load"
    assert b_gs.nelec == gs.nelec, f"GS nelec changed across a save/load: {b_gs.nelec} vs {gs.nelec}"
    assert np.array_equal(b_fch.mo_coeff, fch.mo_coeff), "FCH coefficients changed across a save/load"
    assert np.array_equal(b_fch.mo_energy, fch.mo_energy), "FCH eigenvalues changed across a save/load"
    assert b_xch.e_tot == xch.e_tot, "XCH energy changed across a save/load"
    assert b_exc.orb_idx == exc.orb_idx, "Core orbital index changed across a save/load"
    assert b_exc.channel == ch, f"Excited channel changed across a save/load: {b_exc.channel} vs {ch}"

    for key in exc.mbxas:
        assert np.array_equal(b_exc.mbxas[key], exc.mbxas[key]), \
            f"mbxas['{key}'] changed across a save/load"

    n_before = len(back.excitations)
    back.excite(0)
    assert len(back.excitations) == n_before, "Reloaded object re-ran an excitation it already had"

    energies_rt, absorption_rt, _, _, _ = run_MBXAS_pyscf(
        b_gs.mol, b_gs.to_cpu(), b_fch.to_cpu(), b_exc.orb_idx,
        channel=b_exc.channel, xch_calc=b_xch.to_cpu())

    assert np.allclose(energies_rt, exc.mbxas["energies"], atol=1e-12), \
        "MBXAS re-derived from the checkpoint gives different energies"
    assert np.allclose(absorption_rt, amp_library, atol=1e-12), \
        "MBXAS re-derived from the checkpoint gives different amplitudes"

    spectra_path = str(tmp_path / "spectra.h5")
    spectra = obj.to_spectra()
    spectra.save(spectra_path)
    spectra_back = Spectra.load(spectra_path)

    assert np.array_equal(spectra_back.energies, spectra.energies), \
        "Spectra energies changed across a save/load"
    assert np.array_equal(spectra_back.amplitude, spectra.amplitude), \
        "Spectra amplitudes changed across a save/load"
    assert np.array_equal(spectra_back.CMO, spectra.CMO), \
        "Spectra CMO changed across a save/load"
    assert spectra_back.exc_idx == spectra.exc_idx, "Spectra excited index changed across a save/load"
    assert spectra_back.channel == spectra.channel, "Spectra channel changed across a save/load"
    assert np.array_equal(spectra_back._mb_overlap, spectra._mb_overlap), \
        "Spectra mb_overlap changed across a save/load"
    assert np.array_equal(spectra_back._fch_mo_energy, spectra._fch_mo_energy), \
        "Spectra FCH mo_energy changed across a save/load"
    assert np.array_equal(spectra_back._gs_mo_occ, spectra._gs_mo_occ), \
        "Spectra GS mo_occ changed across a save/load"
    assert spectra_back._core_orb_idx == spectra._core_orb_idx, \
        "Spectra core_orb_idx changed across a save/load"

    egrid_shakeup, kernel_shakeup, orders_shakeup = spectra_fields.get_shakeup_spectrum(order=1, sigma=0.5)
    assert len(egrid_shakeup) == len(kernel_shakeup), "shake-up energy/kernel length mismatch"
    assert np.all(np.isfinite(kernel_shakeup)), "shake-up kernel contains non-finite values"
    assert orders_shakeup is None, \
        "the default overlap diagnostic should combine both spin channels"

    # The streamed diagnostic must equal explicit stick materialization on
    # this small system. Production order-2 diagnostics use the streamed path
    # so their singles x singles product does not need to exist in memory.
    diagnostic_range = [-5.0, 20.0]
    egrid_order2, curve_order2, _ = spectra_fields.get_shakeup_spectrum(
        order=2, channel=spectra_fields._channel, sigma=0.5,
        npoints=501, erange=diagnostic_range, tol=1e-12)
    material_e, material_w, _, _ = spectra_fields._overlap_distribution(
        2, spectator_order=None, channel=spectra_fields._channel,
        tol=1e-12)
    from pymbxas.mbxas.broaden import broadened_spectrum
    assert np.allclose(
        curve_order2,
        broadened_spectrum(egrid_order2, material_e, material_w, 0.5),
        rtol=1e-13, atol=1e-15)

    # explicit channel argument must be accepted (designed-in extension
    # point for the future cross-spin feature), even though only the
    # excited channel is exercised meaningfully here
    _, _, orders_explicit_channel = spectra_fields.get_shakeup_spectrum(order=1, channel=spectra_fields._channel)
    assert orders_explicit_channel == [1]

    # The default f1 spectrum includes the spectator S0 determinant.
    E_none, I_none = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none, E_spectra) and np.array_equal(I_none, I_spectra), \
        "default spin-combined f1 changed get_mbxas_spectra output"

    # f_order=2 adds the explicit f2=20+11 contribution.
    E_sk, I_sk = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, f_order=2)
    assert E_sk.shape == E_none.shape
    assert not np.allclose(I_sk, I_none), \
        "f_order=2 should change the spectrum (H2O/O has nonzero f2 intensity)"
    assert np.all(np.isfinite(I_sk))

    # The spectator Cartesian product is streamed in bounded blocks. A tiny
    # block cap must be algebraically identical to the normal production cap.
    import pymbxas.spectra as spectra_module
    normal_joint_cap = spectra_module._MAX_JOINT_ELEMENTS
    monkeypatch.setattr(spectra_module, "_MAX_JOINT_ELEMENTS", 7)
    E_sk_chunked, I_sk_chunked = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, f_order=2)
    monkeypatch.setattr(spectra_module, "_MAX_JOINT_ELEMENTS", normal_joint_cap)
    assert np.array_equal(E_sk_chunked, E_sk)
    assert np.allclose(I_sk_chunked, I_sk, rtol=1e-13, atol=1e-12)

    # PySCFMBXAS.get_mbxas_spectra must still agree with Spectra's own
    # output after becoming a thin wrapper (extends the existing agreement
    # check above to the f_order path too)
    E_pyscf_sk, I_pyscf_sk = obj.get_mbxas_spectra(
        "O", erange=[520, 560], sigma=0.5, f_order=2)
    assert np.array_equal(E_pyscf_sk, E_sk) and np.allclose(I_pyscf_sk, I_sk, atol=1e-12), \
        "PySCFMBXAS.get_mbxas_spectra(f_order=2) disagrees with Spectra.get_mbxas_spectra"

    # The calculator wrapper must use the same spin-combined f1 default.
    E_pyscf_none, I_pyscf_none = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_pyscf_none, E) and np.allclose(I_pyscf_none, I, atol=1e-12), \
        "PySCFMBXAS.get_mbxas_spectra regression after wrapper refactor"

    # get_mbxas_decomposition must agree with the already-tested individual
    # get_mbxas_spectra/get_shakeup_spectrum calls above
    summary = spectra_fields.get_mbxas_decomposition(
        f_order=3, sigma=0.5, erange=[520, 560])
    assert set(summary["contributions"]) == {1, 2, 3}
    assert set(summary["cumulative"]) == {1, 2, 3}
    assert np.array_equal(summary["energy"], E_none), "get_mbxas_decomposition energy grid mismatch"
    assert np.array_equal(summary["contributions"][1], I_none)
    assert np.array_equal(summary["cumulative"][1], I_none)
    assert np.array_equal(summary["cumulative"][2], I_sk)
    assert np.allclose(
        summary["contributions"][2], I_sk - I_none, atol=1e-15)
    assert set(summary["decomposition"]) == {2, 3}
    for current_f in (2, 3):
        parts = summary["decomposition"][current_f]
        assert set(parts) == {"shakeup", "shakedown"}
        assert np.allclose(
            parts["shakeup"] + parts["shakedown"],
            summary["contributions"][current_f], atol=1e-15)
        assert summary["integrated"]["decomposition"][current_f][
            "shakeup"] + summary["integrated"]["decomposition"][current_f][
                "shakedown"] == pytest.approx(
                    summary["integrated"]["contributions"][current_f])
    assert np.array_equal(summary["total"], summary["cumulative"][3])
    assert summary["integrated"]["cumulative"][1] == pytest.approx(
        np.trapezoid(I_none, E_none)), \
        "get_mbxas_decomposition integrated intensity mismatch"
    prob_e_summary, prob_curve_summary, prob_orders_summary = summary["probability"]
    assert len(prob_e_summary) == len(prob_curve_summary)
    assert prob_orders_summary is None
    overlap_summary = summary["overlap"]
    assert set(overlap_summary["by_total_order"]) == {0, 1, 2}
    assert overlap_summary["captured"] == pytest.approx(
        sum(overlap_summary["by_total_order"].values()))
    assert overlap_summary["available"] >= 0
    assert overlap_summary["captured"] <= (
        overlap_summary["available"] * (1 + 1e-10) + 1e-12)
    if overlap_summary["available"] > 0:
        assert overlap_summary["fraction"] == pytest.approx(
            overlap_summary["captured"] / overlap_summary["available"])

    # plot_mbxas_decomposition is a light smoke test (presentation, not a
    # physics invariant): it should return the right number of Axes
    # without raising, for both the one- and two-panel forms. Force a
    # non-interactive backend first -- this test environment is headless,
    # and a library module should not call matplotlib.use() itself (that
    # is the calling application's decision, not pymbxas's).
    import os
    if not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    from pymbxas.plotting import plot_mbxas_decomposition
    fig_both, axes_both = plot_mbxas_decomposition(summary, show_probability=True)
    assert len(axes_both) == 2, "plot_mbxas_decomposition(show_probability=True) should return 2 axes"
    fig_main, axes_main = plot_mbxas_decomposition(summary, show_probability=False)
    assert len(axes_main) == 1, "plot_mbxas_decomposition(show_probability=False) should return 1 axis"
    import matplotlib.pyplot as _plt
    _plt.close(fig_both)
    _plt.close(fig_main)

    spec_ch2 = 1 - spectra_fields._channel
    spectator_sticks = spectra_fields._spectator_shakeup_sticks(order=1, tol=0.01)
    assert set(spectator_sticks.keys()) <= {1}, \
        f"spectator_shakeup_sticks(order=1) should only include order 1, got {set(spectator_sticks.keys())}"
    spec_e1, spec_w1 = spectator_sticks[1]
    assert np.all(np.isfinite(spec_e1)) and np.all(np.isfinite(spec_w1)), \
        "spectator channel order-1 shake-up sticks contain non-finite values"
    assert np.all(spec_w1 >= 0), "spectator channel shake-up weights must be non-negative"

    # Repeated default calls retain the automatic spectator construction.
    E_none2, I_none2 = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none2, E_none) and np.array_equal(I_none2, I_none), \
        "automatic spectator f1 output changed between calls"
    E_sk2, I_sk2 = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, f_order=2)
    assert np.array_equal(E_sk2, E_sk) and np.array_equal(I_sk2, I_sk), \
        "automatic spectator MB2 output changed between calls"

    # spectator_order alone with f1 must apply a diagnostic correction --
    # a spectator-only cross term reduces to that channel's own shake-up
    E_bare, I_bare = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, spectator_order=None)
    sp_overlap = exc.mbxas["mb_overlap"][spec_ch]
    sp_A = sp_overlap[np.ix_(occ_fch_spec_h, occ_gs_spec_h)]
    E_spec0, I_spec0 = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, spectator_order=0)
    assert np.array_equal(E_spec0, E_bare)
    assert np.allclose(I_spec0, I_bare * abs(np.linalg.det(sp_A)) ** 2,
                       atol=1e-12), \
        "spectator_order=0 must restore the spectator determinant weight"
    assert np.allclose(I_spec0, I_none, atol=1e-12), \
        "the default f1 spectrum must include spectator order zero"
    E_spec_only, I_spec_only = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, spectator_order=1)
    assert E_spec_only.shape == E_bare.shape
    if spec_w1.sum() > 0:
        assert not np.allclose(I_spec_only, I_bare), \
            "spectator_order=1 should change the spectrum when the spectator channel has nonzero shake-up mass"
    assert np.all(np.isfinite(I_spec_only))

    # The combined overlap diagnostic carries determinant weights for both channels.
    de_from_spectra, dw_from_spectra = spectra_fields._combined_shakeup_sticks(1, 1, None, 0.01, False)
    assert np.all(np.isfinite(de_from_spectra)) and np.all(dw_from_spectra >= 0)

    # spectator_order combined with an explicit channel is a conflict --
    # channel identity is fixed by the cross-channel combination itself
    with pytest.raises(ValueError):
        spectra_fields.get_shakeup_spectrum(order=1, channel=spec_ch2, spectator_order=1)

    # PySCFMBXAS.get_mbxas_spectra must forward the new parameters and
    # agree with Spectra's own output, same pattern as the existing
    # f_order agreement check
    E_pyscf_spec, I_pyscf_spec = obj.get_mbxas_spectra(
        "O", erange=[520, 560], sigma=0.5, spectator_order=1)
    assert np.array_equal(E_pyscf_spec, E_spec_only) and np.allclose(I_pyscf_spec, I_spec_only, atol=1e-12), \
        "PySCFMBXAS.get_mbxas_spectra(spectator_order=1) disagrees with Spectra.get_mbxas_spectra"

    real_overlap_diagnostic = spectra_fields._overlap_diagnostic
    diagnostic_calls = 0

    def counted_overlap_diagnostic(*args, **kwargs):
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        return real_overlap_diagnostic(*args, **kwargs)

    monkeypatch.setattr(
        spectra_fields, "_overlap_diagnostic", counted_overlap_diagnostic)

    import pymbxas.mbxas.shakeup as shakeup_module
    real_screened_mb2 = shakeup_module._screened_mb2_sticks
    screening_calls = 0

    def counted_screened_mb2(*args, **kwargs):
        nonlocal screening_calls
        screening_calls += 1
        return real_screened_mb2(*args, **kwargs)

    monkeypatch.setattr(
        shakeup_module, "_screened_mb2_sticks", counted_screened_mb2)
    summary2 = spectra_fields.get_mbxas_decomposition(
        f_order=2, sigma=0.5, erange=[520, 560])
    assert diagnostic_calls == 1, \
        "decomposition should reuse one overlap diagnostic for its curve and mass"
    assert screening_calls == 1, \
        "decomposition should screen MB2 once for full and shake-down spectra"
    assert "shakedown_fraction" in summary2, "get_mbxas_decomposition should report shakedown_fraction"
    assert 0.0 <= summary2["shakedown_fraction"] <= 1.0, \
        f"shakedown_fraction should be a probability fraction in [0, 1], got {summary2['shakedown_fraction']}"
    assert set(summary2["decomposition"]) == {2}
    assert np.allclose(
        summary2["decomposition"][2]["shakeup"]
        + summary2["decomposition"][2]["shakedown"],
        summary2["contributions"][2], atol=1e-15)
    _, independently_selected_down = spectra_fields._get_many_body_mbxas_spectra(
        axis=None, sigma=0.5, npoints=len(summary2["energy"]), tol=0.01,
        erange=[520, 560], max_extra_order=1, spectator_order=1,
        max_total_order=1, shakedown_only=True,
        max_configurations=2_000_000)
    assert np.allclose(
        summary2["decomposition"][2]["shakedown"],
        independently_selected_down, atol=1e-15), \
        "single-pass shake-down accumulation disagrees with direct selection"
    assert spectra_fields.get_mbxas_decomposition(
        f_order=1, sigma=0.5,
        erange=[520, 560])["decomposition"] == {}

    summary_cross = spectra_fields.get_mbxas_decomposition(
        f_order=2, sigma=0.5, erange=[520, 560], spectator_order=1)
    assert set(summary_cross["contributions"]) == {1, 2}
    assert np.array_equal(
        summary_cross["contributions"][1], summary2["contributions"][1])
    assert np.array_equal(summary_cross["total"], summary2["total"]), \
        "automatic and explicit spectator order one should agree"

    E_cross_direct, I_cross_direct = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, f_order=2, spectator_order=1)
    assert np.array_equal(summary_cross["energy"], E_cross_direct)
    assert np.array_equal(summary_cross["total"], I_cross_direct), \
        "summary f2 total should match the direct spin-combined spectrum"
    assert summary_cross["integrated"]["total"] == pytest.approx(
        np.trapezoid(I_cross_direct, E_cross_direct)), \
        "get_mbxas_decomposition's f2 integrated intensity mismatch"

    with pytest.raises(ValueError, match="f_order must be a positive integer"):
        spectra_fields.get_mbxas_spectra(f_order=0)
    with pytest.raises(TypeError, match="shakeup_order"):
        spectra_fields.get_mbxas_spectra(shakeup_order=1)
    with pytest.raises(TypeError, match="shakedown_only"):
        spectra_fields.get_mbxas_spectra(shakedown_only=True)
    with pytest.raises(TypeError, match="shakedown_only"):
        spectra_fields.get_shakeup_spectrum(shakedown_only=True)

    # plot_mbxas_decomposition must handle the spin-combined data.
    fig_cross, axes_cross = plot_mbxas_decomposition(
        summary_cross, show_probability=True)
    assert len(axes_cross) == 2
    _plt.close(fig_cross)

    import io
    report = io.StringIO()
    spectra_fields.print_mbxas_summary(summary2, file=report)
    report_text = report.getvalue()
    assert "MBXAS decomposition through f2" in report_text
    assert "f1 contribution" in report_text
    assert "shake-up" in report_text and "shake-down" in report_text
    assert "overlap shake-down" in report_text
    assert "Overlap convergence" in report_text
    assert "captured fraction" in report_text
    assert not hasattr(spectra_fields, "get_shakeup_summary")
    import pymbxas.plotting as plotting_module
    assert not hasattr(plotting_module, "plot_shakeup_summary")
