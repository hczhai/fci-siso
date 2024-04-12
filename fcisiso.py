
#  fci-siso : State Interaction Spin-Orbit (SISO) Method for CASSCF and FCI
#
#  Copyright (C) 2021 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pyscf import fci, mcscf
from pyscf.data import nist
import numpy as np
import copy


def _print_matrix(mat):
    for mm in mat:
        print("".join(["%9.5f" % x for x in mm]))
    print('-' * 60)


def get_jk(mol, dm0):
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(
        3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk += np.einsum('yijkl,li->ykj', hso2e, dm0)
    return vj, vk


def get_jk_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def compute_hso_ao(mol, dm0, qed_fac=1, amfi=False):
    """hso (complex, pure imag)"""
    alpha2 = nist.ALPHA ** 2
    hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    vj, vk = get_jk_amfi(mol, dm0) if amfi else get_jk(mol, dm0)
    hso2e = vj - vk * 1.5
    hso = qed_fac * (alpha2 / 4) * (hso1e + hso2e)
    if mol.has_ecp_soc():
        hso -= 0.5 * mol.intor("ECPso")
    return hso * 1j


# dm_pq = <|qs^+ pr|>; (s, r) = dspin
# note that pyscf pdm1 def is strange!!!
# 1pdm[p,q] = :math:`\langle q^\dagger p \rangle`
def make_trans_rdm1(dspin, cibra, ciket, norb, nelec_bra, nelec_ket):
    """
    One-particle transition density matrix between states
    with different spins.

    Args:
        dspin : str 'aa' or 'bb' or 'ab' or 'ba'
            the spin subscript of p, q operators
        cibra : np.ndarray((n_det_alpha, n_det_beta))
            ci vector for bra wavefunction
        ciket : np.ndarray((n_det_alpha, n_det_beta))
            ci vector for ket wavefunction
        norb : int
            number of orbitals
        nelec_bra : (int, int)
            numebr of alpha and beta electrons in bra state
        nelec_ket : (int, int)
            numebr of alpha and beta electrons in ket state

    Returns:
        rdm1 : np.ndarray((norb, norb))
            transition density matrix
    """
    nelabra, nelbbra = nelec_bra
    nelaket, nelbket = nelec_ket
    if dspin == 'ba':
        cond = nelabra == nelaket - 1 and nelbbra == nelbket + 1
    elif dspin == 'ab':
        cond = nelabra == nelaket + 1 and nelbbra == nelbket - 1
    elif dspin == 'aa':
        cond = nelabra == nelaket and nelbbra == nelbket and nelabra > 0
    else:
        cond = nelabra == nelaket and nelbbra == nelbket and nelbbra > 0
    if not cond:
        return np.array(0)
    nabra = fci.cistring.num_strings(norb, nelabra)
    nbbra = fci.cistring.num_strings(norb, nelbbra)
    naket = fci.cistring.num_strings(norb, nelaket)
    nbket = fci.cistring.num_strings(norb, nelbket)
    cibra = cibra.reshape(nabra, nbbra)
    ciket = ciket.reshape(naket, nbket)
    lidxbra = fci.cistring.gen_des_str_index(range(norb),
                                             nelabra if dspin[0] == 'a' else nelbbra)
    if dspin[1] == 'a':
        lidxket = fci.cistring.gen_des_str_index(range(norb), nelaket)
        naketd = fci.cistring.num_strings(norb, nelaket - 1)
        t1 = np.zeros((norb, naketd, nbket))
        for str0 in range(naket):
            for _, i, str1, sign in lidxket[str0]:
                t1[i, str1, :] += sign * ciket[str0, :]
    else:
        lidxket = fci.cistring.gen_des_str_index(range(norb), nelbket)
        nbketd = fci.cistring.num_strings(norb, nelbket - 1)
        t1 = np.zeros((norb, naket, nbketd))
        for str0 in range(nbket):
            for _, i, str1, sign in lidxket[str0]:
                t1[i, :, str1] += sign * ciket[:, str0]
        if nelaket % 2 == 1:
            t1 = -t1
    if dspin[0] == 'a':
        lidxbra = fci.cistring.gen_des_str_index(range(norb), nelabra)
        nabrad = fci.cistring.num_strings(norb, nelabra - 1)
        t2 = np.zeros((norb, nabrad, nbbra))
        for str0 in range(nabra):
            for _, i, str1, sign in lidxbra[str0]:
                t2[i, str1, :] += sign * cibra[str0, :]
    else:
        lidxbra = fci.cistring.gen_des_str_index(range(norb), nelbbra)
        nbbrad = fci.cistring.num_strings(norb, nelbbra - 1)
        t2 = np.zeros((norb, nabra, nbbrad))
        for str0 in range(nbbra):
            for _, i, str1, sign in lidxbra[str0]:
                t2[i, :, str1] += sign * cibra[:, str0]
        if nelabra % 2 == 1:
            t2 = -t2
    rdm1 = np.tensordot(t1, t2, axes=((1, 2), (1, 2)))
    return rdm1


def make_trans(m, cibra, ciket, norb, nelec_bra, nelec_ket):
    """
    J. Chem. Theory Comput. 2016, 12, 5881−5894
    Eq. (54-56)
    """
    if m == 1:
        return -1.0 * make_trans_rdm1('ab', cibra, ciket, norb, nelec_bra, nelec_ket).T
    elif m == -1:
        return make_trans_rdm1('ba', cibra, ciket, norb, nelec_bra, nelec_ket).T
    else:
        return np.sqrt(0.5) * (make_trans_rdm1('aa', cibra, ciket, norb, nelec_bra, nelec_ket)
                               - make_trans_rdm1('bb', cibra, ciket, norb, nelec_bra, nelec_ket)).T


def extract_ci_list(mc, tol=1e-3):
    from pyscf.mcscf.addons import StateAverageMixFCISolver
    if isinstance (mc.fcisolver, StateAverageMixFCISolver):
        solver_list = mc.fcisolver.fcisolvers
    else:
        solver_list = [mc.fcisolver]
    ci_list = []
    cum_root = 0
    for i_solver in solver_list:
        nstates = i_solver.nstates
        na, nb = i_solver.nelec
        for j_root in range(nstates):
            (j_s_square, j_mult) = i_solver.spin_square(mc.ci[cum_root + j_root],
                                                        i_solver.norb, i_solver.nelec)
            j_2s = round(j_mult - 1)
            if abs(np.rint(j_s_square)-j_s_square) > tol:
                print(f"spin contamination S~{j_2s/2}, S^2={j_s_square} "
                      + f"for root {cum_root + j_root}")
            ci_list.append((na, nb, j_2s, na-nb, mc.e_states[cum_root + j_root],
                            mc.ci[cum_root + j_root]))
        cum_root += nstates
    return ci_list


class FCISISO:
    """
    FCI state-interaction for spin-orbit coupling.

    Args:
        mol : Mole object
        mf : scf object (FCI mode) or mcscf object (CASCI mode)
        states : list((nroots, multiplicity, wfnsym : str))
            or list((nroots, multiplicity))
            states to include
        weights : list(tuple(float)) or None
            weight for each state (only useful for CASCI mode)
        cas : (norb, nelec) (CASCI mode) or None (FCI mode)
            active space for CASCI
    
    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom="O 0 0 0; H 0.76 0 0.5; H 0.76 0 -0.5")
    >>> mol.symmetry = True
    >>> mol.basis = 'sto-3g'
    >>> mol.build()
    >>> mf = scf.RHF(mol).run()
    converged SCF energy = -74.8769700837748
    >>> from fcisiso import FCISISO
    >>> states = [(4, 1, 'A1'), (4, 3, 'B1'), (4, 3, 'A2'), (4, 3, 'B2')]
    >>> FCISISO(mol, mf, states).kernel()
    array([-74.92867132, -74.51038116, -74.51038108,
        ..., -73.60940709, -73.60940704])
    """

    def __init__(self, mol, mf, states=None, weights=None, cas=None):
        self.mol = mol
        self.mo_coeff = mf.mo_coeff
        self.cas = cas
        if cas is None:
            self.ff = fci.FCI(mol, mf.mo_coeff)
            self.ncore = 0
            self.norb = mol.nao
        else:
            self.ff = mcscf.CASCI(mf, cas[0], cas[1])
            self.ncore = self.ff.ncore
            self.norb = cas[0]
            self.weights = weights
        self.ci = None
        self.states = states

    def kernel(self, hsoao=None, dmao=None, amfi=True):
        """
        State-Interaction.

        If self.ci is None, the states will be calculated using FCI or CASCI
        self.ci can also be set manually before calling this method,
        to skip the FCI or CASCI step.

        self.ci : list((na, nb, 2S, 2MS, energy, ci vector))
            where :
                na : number of alpha electrons
                nb : number of beta electrons
                2S : 2 times total spin
                2MS : 2 times projected spin
                ci vector : 2-dim np.array with shape
                    (n_det_alpha, n_det_beta)

        Args:
            hsoao : np.ndarray((3, nao, nao)) or None
                Hso integral in AO basis
                if None, it will be calculated using pyscf

            dmao : np.ndarray((n_all_orbs, n_all_orbs)) or None
                1pdm of ground state, in AO basis
                used for calculating hsoao

            amfi : bool
                whether amfi should be used for calculating hsoao
                for large number of atoms, using amfi can significantly
                save hsoao computing time
                if amfi == False, a full
                np.ndarray((3, nao, nao, nao, nao)) is generated as
                an intermediate.

        Returns:
            list(float) : Energy eigenstates with SOC
        """
        if self.ci is None or len(self.ci) == 0:
            assert self.states is not None
            if self.cas is None:
                nelec = self.mol.nelec[0] + self.mol.nelec[1]
            else:
                nelec = self.cas[1]
                if not isinstance(nelec, (int, np.integer)):
                    nelec = nelec[0] + nelec[1]
            print('\nCalculate spin-free eigenstates:\n')
            self.ci = []
            for ist, state in enumerate(self.states):
                nroots = state[0]
                mult = state[1]
                wfnsym = state[2] if len(state) == 3 else None
                s2 = mult - 1
                if self.cas is None:
                    fg = fci.addons.fix_spin_(
                        self.ff, ss=s2 / 2 * (s2 / 2 + 1))
                else:
                    self.ff.fix_spin_(ss=s2 / 2 * (s2 / 2 + 1))
                for ms2 in range(-s2, s2 + 1, 2):
                    assert (nelec + ms2) % 2 == 0
                    na = (nelec + ms2) // 2
                    nb = nelec - na
                    ciidxs = []
                    ext_nroots = nroots
                    while len(ciidxs) < nroots:
                        print('NR = %2d S = %3.1f MS = %4.1f NA = %2d NB = %2d SYM = %r' %
                              (ext_nroots, s2 / 2, ms2 / 2, na, nb, wfnsym), end='')
                        if self.cas is not None:
                            fg = copy.copy(self.ff)
                            fg.nelecas = (na, nb)
                            if self.weights is not None:
                                fg.state_average_(weights=self.weights[ist])
                            else:
                                fg.state_average_(
                                    weights=[1.0 / ext_nroots] * ext_nroots)
                            fg.fcisolver.wfnsym = wfnsym
                            fg.kernel(self.mo_coeff)
                            e = fg.e_states
                            ci = fg.ci
                            ciidxs = list(range(0, ext_nroots))
                        else:
                            e, ci = fg.kernel(nroots=ext_nroots,
                                              wfnsym=wfnsym, nelec=(na, nb))
                            sps = np.array(
                                [fg.spin_square(cci, self.mol.nao, (na, nb))[1] for cci in ci])
                            ciidxs = list(np.argwhere(
                                np.abs(sps - mult) < 1E-4).reshape(-1))
                        print('  Energies = ',
                              * ["%15.8f" % x for x in e[ciidxs]])
                        ext_nroots += 1
                    ciidxs = ciidxs[:nroots]
                    for ir in range(nroots):
                        self.ci.append(
                            (na, nb, s2, ms2, e[ciidxs[ir]], ci[ciidxs[ir]]))
        if hsoao is None:
            print('\nGenerating Spin-Orbit Integrals:\n')
            gsci = 0
            for ici, ci in enumerate(self.ci):
                if ci[4] < self.ci[gsci][4]:
                    gsci = ici
            if dmao is None:
                dmao = self.ff.make_rdm1(
                    self.ci[gsci][-1], self.mol.nao, self.ci[gsci][:2])
                if self.cas is None:
                    # for FCI solver, transform dm1 from mo to ao
                    dmao = self.mo_coeff @ dmao @ self.mo_coeff.T
            hsoao = compute_hso_ao(self.mol, dmao, amfi=amfi) * 2
        hso = np.einsum('rij,ip,jq->rpq', hsoao,
                        self.mo_coeff[:, self.ncore:self.ncore + self.norb],
                        self.mo_coeff[:, self.ncore:self.ncore + self.norb])
        print(np.linalg.norm(hso))
        # state interaction
        hdiag = np.array([ci[4] for ci in self.ci], dtype=complex)
        hsiso = np.zeros((len(self.ci), len(self.ci)), dtype=complex)
        thrds = 29.0  # cm-1
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        print("\nComplex SO-Hamiltonian matrix elements over spin components of spin-free eigenstates:")
        print("(In cm-1. Print threshold:%10.3f cm-1)\n" % thrds)
        for istate, ici in enumerate(self.ci):
            for jstate, jci in enumerate(self.ci):
                if jstate > istate:
                    continue
                tp1 = make_trans(1, ici[-1], jci[-1],
                                 self.norb, ici[:2], jci[:2])
                tze = make_trans(0, ici[-1], jci[-1],
                                 self.norb, ici[:2], jci[:2])
                tm1 = make_trans(-1, ici[-1], jci[-1],
                                 self.norb, ici[:2], jci[:2])
                t = np.zeros((3, self.norb, self.norb), dtype=complex)
                t[0] = (0.5 + 0j) * (tm1 - tp1)
                t[1] = (0.5j + 0) * (tm1 + tp1)
                t[2] = (np.sqrt(0.5) + 0j) * tze
                somat = np.einsum('rij,rij->', t, hso)
                hsiso[istate, jstate] = somat
                if istate != jstate:
                    hsiso[jstate, istate] = somat.conj()
                somat *= au2cm
                if abs(somat) > thrds:
                    print(('  I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f '
                           + 'I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f Re = %9.3f Im = %9.3f')
                          % (istate, ici[4], ici[2] / 2, ici[3] / 2, jstate, jci[4], jci[2] / 2,
                             jci[3] / 2, somat.real, somat.imag))
        # Full SISO Hamiltonian eigen states
        hfull = hsiso + np.diag(hdiag)
        heig, hvec = np.linalg.eigh(hfull)
        print('\nTotal energies including SO-coupling:\n')
        for i in range(len(heig)):
            shvec = []
            ess = []
            imb = 0
            if self.states is None:
                shvec = np.abs(hvec[:, i] ** 2)
                ess = [(vec[4], vec[2]) for vec in self.ci]
            else:
                for ibra in range(len(self.states)):
                    mult = self.states[ibra][1]
                    nr = self.states[ibra][0]
                    for ir in range(nr):
                        shvec.append(np.linalg.norm(
                            hvec[imb + ir:imb + ir + mult * nr:nr, i]) ** 2)
                        ess.append((self.ci[imb + ir][4], mult - 1))
                    imb += mult * nr
                assert imb == len(heig)
            iv = np.argmax(np.abs(shvec))
            print('  State %4d Total energy: %15.8f | largest |coeff|**2 %10.6f from I = %4d E = %15.8f S = %4.1f'
                  % (i, heig[i], shvec[iv], iv, ess[iv][0], ess[iv][1] / 2))
        self.energies = heig
        return heig
