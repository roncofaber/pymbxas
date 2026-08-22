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
        target_dir=str(tmp_path)
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
    assert abs(det_A - 0.9486) < 0.05, f"det(A) = {det_A:.4f}, expected ~0.9486"

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

    # order=2: weight is the antisymmetrized 2x2 minor of K, matching
    # mbxas-qe's doubles_overlap formula exactly (K(v,c)*K(vp,cp)-K(v,cp)*K(vp,c))
    e2, w2 = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    v0, v1_ = 0, 1
    c0, c1_ = 0, 1
    manual_minor = K_ch[v0, c0] * K_ch[v1_, c1_] - K_ch[v0, c1_] * K_ch[v1_, c0]
    assert any(abs(w - abs(manual_minor) ** 2) < 1e-14 for w in w2), \
        "no order=2 stick matches the hand-computed 2x2 minor for the first valence/conduction pair"

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
