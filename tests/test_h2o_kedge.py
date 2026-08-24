import logging

import numpy as np
import pytest
import ase.build
from ase import units
from pymbxas.calculators.pyscf import PySCF_mbxas
from pymbxas.build.structure import ase_to_mole


def test_h2o_oxygen_kedge(tmp_path):
    structure = ase.build.molecule("H2O")

    obj = PySCF_mbxas(
        structure=structure,
        charge=0,
        spin=0,
        xc="lda",
        basis="def2-svpd",
        calc_type="UKS",
        loc_type="ibo",
        xas_verbose=1,
        dft_verbose=0,
        dft_output=False,
        save=False,
        target_dir=str(tmp_path),
        gpu=True,
    )

    obj.kernel("O")

    exc = obj.excitations[0]
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

    # ase_to_mole must warn when it forwards an unrecognized kwarg to
    # gto.Mole -- attach a handler directly to the module logger rather
    # than relying on propagation to a root-attached capture handler,
    # since configure_logger() sets propagate=False on the "pymbxas" logger
    class _RecordCollector(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []
        def emit(self, record):
            self.records.append(record)

    collector = _RecordCollector()
    structure_logger = logging.getLogger("pymbxas.build.structure")
    structure_logger.addHandler(collector)
    structure_logger.setLevel(logging.WARNING)
    try:
        ase_to_mole(st2, 0, 0, basis="def2-svpd", pbc=False, verbose=0,
                    print_output=False, this_kwarg_does_not_exist=True)
    finally:
        structure_logger.removeHandler(collector)
    assert any("forwarding unrecognized keyword" in r.getMessage() for r in collector.records), \
        "ase_to_mole should warn when forwarding an unrecognized kwarg to gto.Mole"

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
    spectra_direct = obj.to_spectra(0)
    assert np.allclose(spectra_direct.amp2int(), per_transition_intensity, atol=1e-15), \
        "Spectra.amp2int() does not include the omega prefactor"

    # PySCF_mbxas.get_mbxas_spectra and Spectra.get_mbxas_spectra must stay
    # numerically identical (see dev/method.md gotcha on the three
    # get_mbxas_spectra implementations).
    E_spectra, I_spectra = spectra_direct.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.allclose(E, E_spectra) and np.allclose(I, I_spectra, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra and Spectra.get_mbxas_spectra disagree"

    assert amp_library.shape[0] == 3, f"Amplitude first dimension should be 3 (Cartesian), got {amp_library.shape[0]}"
    assert amp_library.shape[1] == len(exc.mbxas["energies"]), f"Amplitude transitions mismatch: {amp_library.shape[1]} vs {len(exc.mbxas['energies'])}"

    spectra_fields = obj.to_spectra(0)
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
    _, _, K_ch = build_A_K(mb_overlap_ch, occ_idxs_fch_ch, occ_idxs_gs_ch, uno_idxs_fch_ch)
    eps_occ_ch = fch.mo_energy[ch][occ_idxs_fch_ch]
    eps_unocc_ch = fch.mo_energy[ch][uno_idxs_fch_ch]

    # order=1 shake-up recovers a plain |K_vc|^2 stick spectrum, one entry
    # per (valence, conduction) pair
    e1, w1 = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert e1.shape == w1.shape == (len(occ_idxs_fch_ch) * len(uno_idxs_fch_ch),), \
        f"order=1 shake-up stick count mismatch: {e1.shape} vs expected {(len(occ_idxs_fch_ch)*len(uno_idxs_fch_ch),)}"
    w1_manual = np.abs(K_ch) ** 2
    assert np.allclose(np.sort(w1), np.sort(w1_manual.ravel()), atol=1e-14), \
        "order=1 shake-up weights do not match |K_vc|^2"

    # shakedown_only filters to negative delta_e only ("shake-down",
    # mbxas-qe's kpoint_spectral_details.f90 convention), at the
    # single-order level
    e1_down, w1_down = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.all(e1_down < 0), "shakedown_only=True should keep only negative delta_e sticks"
    manual_mask = e1 < 0
    assert np.array_equal(np.sort(e1_down), np.sort(e1[manual_mask])), \
        "shakedown_only=True should match a manual delta_e<0 filter of the unfiltered order-1 sticks"
    assert np.array_equal(np.sort(w1_down), np.sort(w1[manual_mask])), \
        "shakedown_only=True should keep the matching weights unchanged"

    # order=2: weight is the antisymmetrized 2x2 minor of K, matching
    # mbxas-qe's doubles_overlap formula exactly. K has shape (n_unocc, n_occ),
    # so K[c,v] = K[conduction_idx, valence_idx]. Use non-degenerate indices
    # (valence {0,1}, conduction {0,2}) to verify correct axis assignment.
    e2, w2 = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    v0, v1_ = 0, 1
    c0, c1_ = 0, 2
    manual_minor = K_ch[c0, v0] * K_ch[c1_, v1_] - K_ch[c0, v1_] * K_ch[c1_, v0]
    assert any(abs(w - abs(manual_minor) ** 2) < 1e-14 for w in w2), \
        "no order=2 stick matches the hand-computed 2x2 minor for valence pair (0,1) and conduction pair (0,2)"

    # order=3 is explicitly out of scope for this version
    with pytest.raises(NotImplementedError):
        shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=3)

    # shakeup_spectrum: explicit order=1 includes only order 1; explicit
    # order=2 always includes both orders (no silent auto-downgrade)
    de1, dw1, orders1 = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert orders1 == [1], f"explicit order=1 should include only order 1, got {orders1}"
    de2, dw2, orders2 = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders2 == [1, 2], f"explicit order=2 should include orders [1, 2], got {orders2}"
    assert len(de2) == len(e1) + len(e2), "order=2 spectrum should concatenate order-1 and order-2 sticks"

    # auto mode never includes an order whose total probability mass is
    # below tol * order-1 mass; physically, higher-order shake-up should
    # carry less total probability than order 1
    assert w2.sum() < w1.sum(), \
        f"order-2 total shake-up probability ({w2.sum():.3e}) should be smaller than order-1 ({w1.sum():.3e})"
    de_auto, dw_auto, orders_auto = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order="auto", tol=0.01)
    assert orders_auto in ([1], [1, 2]), f"auto order resolved to unexpected {orders_auto}"
    if w2.sum() > 0.01 * w1.sum():
        assert orders_auto == [1, 2]
    else:
        assert orders_auto == [1]

    from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

    sticks_by_order_2, orders_by_order_2 = shakeup_sticks_by_order(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders_by_order_2 == [1, 2], f"expected orders [1, 2], got {orders_by_order_2}"
    assert np.array_equal(sticks_by_order_2[1][0], e1) and np.array_equal(sticks_by_order_2[1][1], w1), \
        "shakeup_sticks_by_order order-1 entry should match shakeup_sticks(order=1)"
    assert np.array_equal(sticks_by_order_2[2][0], e2) and np.array_equal(sticks_by_order_2[2][1], w2), \
        "shakeup_sticks_by_order order-2 entry should match shakeup_sticks(order=2)"

    de2_from_dict = np.concatenate([sticks_by_order_2[k][0] for k in orders_by_order_2])
    dw2_from_dict = np.concatenate([sticks_by_order_2[k][1] for k in orders_by_order_2])
    assert np.array_equal(de2_from_dict, de2) and np.array_equal(dw2_from_dict, dw2), \
        "shakeup_spectrum(order=2) must equal the concatenation of shakeup_sticks_by_order's entries"

    sticks_by_order_down, _ = shakeup_sticks_by_order(K_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
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
    assert np.array_equal(de_solo, sticks_a[1][0]) and np.array_equal(dw_solo, sticks_a[1][1]), \
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
    # unchanged (this is the shakeup_order=None-equivalent limit)
    main_probe = np.exp(-0.5 * (egrid_probe / 1.0) ** 2)
    convolved_empty = convolve_shakeup(egrid_probe, main_probe, np.empty(0), np.empty(0), sigma=0.5)
    assert np.allclose(convolved_empty, main_probe, atol=1e-3), \
        "convolving with an empty shake-up spectrum should not change the main spectrum"

    # a single shake-up stick at a known offset should shift probability
    # mass to (roughly) that offset, and total integrated intensity should
    # be conserved (both terms sum to the original mass, up to the
    # normalization convention: main-only weight 1 vs shake-up weight w)
    stick_de = np.array([2.0])
    stick_w = np.array([1.0])  # equal weight to the n=0 term, for an easy 50/50 check
    convolved_one = convolve_shakeup(egrid_probe, main_probe, stick_de, stick_w, sigma=0.5)
    assert np.trapezoid(convolved_one, egrid_probe) == pytest.approx(
        np.trapezoid(main_probe, egrid_probe), rel=0.05), \
        "convolution should conserve total integrated intensity"
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

    # get_shakeup_spectrum's default erange must widen on the negative side
    # too, not just the positive one -- a negative shake-up stick is
    # possible for a non-aufbau MOM-converged state, not exercised by
    # H2O/O itself, so seed the cache directly with a synthetic one and
    # confirm the real method's erange logic covers it
    if not hasattr(spectra_fields, "_shakeup_cache"):
        spectra_fields._shakeup_cache = {}
    synth_key = (spectra_fields._channel, "synthetic_negative_test", 0.01)
    spectra_fields._shakeup_cache[synth_key] = (np.array([-30.0, 5.0]), np.array([0.5, 0.5]), [1])
    E_synth, I_synth, _ = spectra_fields.get_shakeup_spectrum(order="synthetic_negative_test", sigma=0.5)
    assert E_synth.min() < -30.0, \
        f"erange should extend past the negative stick at -30 eV, got min {E_synth.min():.2f}"
    assert E_synth.max() >= 5.0 + 5 * 0.5, \
        f"erange should still extend past the positive stick, got max {E_synth.max():.2f}"

    # verbosity level 5 must configure the pymbxas logger to a strictly
    # more detailed level than 4 (previously both mapped to logging.DEBUG,
    # making them indistinguishable)
    from pymbxas.io.config import configure_logger, TRACE
    assert TRACE < logging.DEBUG, f"TRACE ({TRACE}) should be below DEBUG ({logging.DEBUG})"
    configure_logger(5)
    assert logging.getLogger("pymbxas").level == TRACE, \
        "verbosity level 5 should configure the pymbxas logger to TRACE, not DEBUG"
    configure_logger(3)  # restore a normal level so later output isn't flooded

    h5_path = obj.save_object(oname="roundtrip.h5", save_path=str(tmp_path))

    from pyscf.scf import chkfile as pyscf_chkfile
    from pymbxas.calculators.pyscf import PySCF_mbxas as _PySCF_mbxas
    from pymbxas.mbxas.mbxas import run_MBXAS_pyscf
    from pymbxas.spectra import Spectra

    mol_chk, scf_chk = pyscf_chkfile.load_scf(h5_path)
    assert mol_chk.natm == 3, "Checkpoint is not readable as a PySCF chkfile"
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(gs.mo_coeff)), \
        "chkfile-read GS coefficients differ from the in-memory ones"

    back = _PySCF_mbxas.load(h5_path)
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
    assert orders_shakeup == [1], f"expected orders [1], got {orders_shakeup}"

    # caching: a second call with the same (channel, order, tol) must reuse
    # the cached sticks rather than recomputing (same object identity)
    cache_key = (spectra_fields._channel, 1, 0.01)
    assert cache_key in spectra_fields._shakeup_cache, "shake-up sticks were not cached"
    cached_before = spectra_fields._shakeup_cache[cache_key]
    spectra_fields.get_shakeup_spectrum(order=1, sigma=0.7)  # different sigma, same order/channel/tol
    assert spectra_fields._shakeup_cache[cache_key] is cached_before, \
        "changing sigma should not invalidate the cached shake-up sticks"

    # explicit channel argument must be accepted (designed-in extension
    # point for the future cross-spin feature), even though only the
    # excited channel is exercised meaningfully here
    _, _, orders_explicit_channel = spectra_fields.get_shakeup_spectrum(order=1, channel=spectra_fields._channel)
    assert orders_explicit_channel == [1]

    # shakeup_order=None (the default) must be byte-identical to the
    # existing get_mbxas_spectra call already exercised above
    E_none, I_none = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none, E_spectra) and np.array_equal(I_none, I_spectra), \
        "shakeup_order=None changed get_mbxas_spectra output"

    # shakeup_order=1 must change the spectrum shape (correction is applied)
    # but conserve total integrated intensity, since convolution conserves
    # the integral of a unit-normalized kernel
    E_sk, I_sk = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert E_sk.shape == E_none.shape
    assert not np.allclose(I_sk, I_none), \
        "shakeup_order=1 should change the spectrum (H2O/O has nonzero order-1 shake-up mass)"
    assert np.trapezoid(I_sk, E_sk) == pytest.approx(np.trapezoid(I_none, E_none), rel=0.1), \
        "shake-up convolution should approximately conserve total integrated intensity within the plotted erange"

    # PySCF_mbxas.get_mbxas_spectra must still agree with Spectra's own
    # output after becoming a thin wrapper (extends the existing agreement
    # check above to the shakeup_order path too)
    E_pyscf_sk, I_pyscf_sk = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert np.array_equal(E_pyscf_sk, E_sk) and np.allclose(I_pyscf_sk, I_sk, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra(shakeup_order=1) disagrees with Spectra.get_mbxas_spectra"

    # shakeup_order=None through PySCF_mbxas must still match the original
    # pre-refactor baseline computed at the top of this test
    E_pyscf_none, I_pyscf_none = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_pyscf_none, E) and np.allclose(I_pyscf_none, I, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra regression after wrapper refactor"

    # get_shakeup_summary must agree with the already-tested individual
    # get_mbxas_spectra/get_shakeup_spectrum calls above
    summary = spectra_fields.get_shakeup_summary(order=2, sigma=0.5, erange=[520, 560])
    assert set(summary["spectra"].keys()) == {0, 1, 2}, \
        f"get_shakeup_summary should return spectra for orders 0,1,2, got {set(summary['spectra'].keys())}"
    assert np.array_equal(summary["energy"], E_none), "get_shakeup_summary energy grid mismatch"
    assert np.array_equal(summary["spectra"][0], I_none), "get_shakeup_summary order-0 (bare) spectrum mismatch"
    assert np.array_equal(summary["spectra"][1], I_sk), "get_shakeup_summary order-1 spectrum mismatch"
    assert summary["integrated"][0] == pytest.approx(np.trapezoid(I_none, E_none)), \
        "get_shakeup_summary integrated intensity mismatch"
    prob_e_summary, prob_curve_summary, prob_orders_summary = summary["probability"]
    assert len(prob_e_summary) == len(prob_curve_summary)
    assert prob_orders_summary in ([1], [1, 2])

    # plot_shakeup_summary is a light smoke test (presentation, not a
    # physics invariant): it should return the right number of Axes
    # without raising, for both the one- and two-panel forms. Force a
    # non-interactive backend first -- this test environment is headless,
    # and a library module should not call matplotlib.use() itself (that
    # is the calling application's decision, not pymbxas's).
    import os
    if not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    from pymbxas.plotting import plot_shakeup_summary
    fig_both, axes_both = plot_shakeup_summary(summary, show_probability=True)
    assert len(axes_both) == 2, "plot_shakeup_summary(show_probability=True) should return 2 axes"
    fig_main, axes_main = plot_shakeup_summary(summary, show_probability=False)
    assert len(axes_main) == 1, "plot_shakeup_summary(show_probability=False) should return 1 axis"
    import matplotlib.pyplot as _plt
    _plt.close(fig_both)
    _plt.close(fig_main)

    from pymbxas.mbxas.shakeup import combine_cross_channel_sticks as _combine_cc

    spec_ch2 = 1 - spectra_fields._channel
    spectator_sticks = spectra_fields._spectator_shakeup_sticks(order=1, tol=0.01)
    assert set(spectator_sticks.keys()) <= {1}, \
        f"spectator_shakeup_sticks(order=1) should only include order 1, got {set(spectator_sticks.keys())}"
    spec_e1, spec_w1 = spectator_sticks[1]
    assert np.all(np.isfinite(spec_e1)) and np.all(np.isfinite(spec_w1)), \
        "spectator channel order-1 shake-up sticks contain non-finite values"
    assert np.all(spec_w1 >= 0), "spectator channel shake-up weights must be non-negative"

    # spectator_order=None/max_total_order=None must remain byte-identical
    # to the pre-cross-spin behavior already exercised above
    E_none2, I_none2 = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none2, E_none) and np.array_equal(I_none2, I_none), \
        "spectator_order=None regression: get_mbxas_spectra output changed"
    E_sk2, I_sk2 = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert np.array_equal(E_sk2, E_sk) and np.array_equal(I_sk2, I_sk), \
        "spectator_order=None regression: get_mbxas_spectra(shakeup_order=1) output changed"

    # spectator_order alone (shakeup_order=None) must apply a correction --
    # a spectator-only cross term reduces to that channel's own shake-up
    E_bare, I_bare = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    E_spec_only, I_spec_only = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, spectator_order=1)
    assert E_spec_only.shape == E_bare.shape
    if spec_w1.sum() > 0:
        assert not np.allclose(I_spec_only, I_bare), \
            "spectator_order=1 should change the spectrum when the spectator channel has nonzero shake-up mass"
    assert np.trapezoid(I_spec_only, E_spec_only) == pytest.approx(
        np.trapezoid(I_bare, E_bare), rel=0.15), \
        "spectator-only shake-up convolution should approximately conserve total integrated intensity"

    # combining both channels must agree with a manual combine_cross_channel_sticks call
    excited_sticks_by_order = spectra_fields._shakeup_sticks_by_order(1, None, 0.01)
    de_manual, dw_manual = _combine_cc(excited_sticks_by_order, spectator_sticks, max_total_order=2)
    de_from_spectra, dw_from_spectra = spectra_fields._combined_shakeup_sticks(1, 1, None, 0.01, False)
    assert np.array_equal(np.sort(de_from_spectra), np.sort(de_manual)) and \
           np.allclose(np.sort(dw_from_spectra), np.sort(dw_manual), atol=1e-15), \
        "_combined_shakeup_sticks(shakeup_order=1, spectator_order=1) disagrees with a manual combine_cross_channel_sticks call"

    # spectator_order combined with an explicit channel is a conflict --
    # channel identity is fixed by the cross-channel combination itself
    with pytest.raises(ValueError):
        spectra_fields.get_shakeup_spectrum(order=1, channel=spec_ch2, spectator_order=1)

    # PySCF_mbxas.get_mbxas_spectra must forward the new parameters and
    # agree with Spectra's own output, same pattern as the existing
    # shakeup_order agreement check
    E_pyscf_spec, I_pyscf_spec = obj.get_mbxas_spectra(
        "O", erange=[520, 560], sigma=0.5, spectator_order=1)
    assert np.array_equal(E_pyscf_spec, E_spec_only) and np.allclose(I_pyscf_spec, I_spec_only, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra(spectator_order=1) disagrees with Spectra.get_mbxas_spectra"

    # shakedown_only must not raise and must never increase total mass
    E_shakedown, I_shakedown = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, shakeup_order=1, shakedown_only=True)
    assert np.all(np.isfinite(I_shakedown))
