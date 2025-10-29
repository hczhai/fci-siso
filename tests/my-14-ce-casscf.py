import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf
from fcisiso import FCISISO
import numpy as np

mol = gto.M(atom='Ce 0 0 0',
            symmetry=False,
            basis='ccpvtz-dk', spin=1, charge=3, verbose=3)
mf = scf.newton(scf.RHF(mol).sfx2c1e())
mf.kernel()

print("\n Molecular orbital components:\n")

# total: 55 elec
# core: 1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 (54e, 27o)
# active: 4f1 (1e, 7o)
ao_lb = gto.sph_labels(mol)
for i in range(40):
    ivs = np.argsort(mf.mo_coeff[:, i]**2)
    ss = ""
    for iv in ivs[::-1][:3]:
        ss += "%10.6f (%s)" % (mf.mo_coeff[iv, i] ** 2, ao_lb[iv].strip())
    print(" MO %2d :: %s" % (i, ss))

print("\n CASSCF(1, 7):\n")

cas_list = [28, 29, 30, 31, 32, 33, 34]
mc = mcscf.CASSCF(mf, 7, 1)
mc.fix_spin_(ss=0.75)
mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list, base=0)
mc.state_average_((1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7))
mc.kernel(mo)

print(mc.nelecas, mc.e_states)

dmao = mc.make_rdm1()
print(dmao.shape)

print("\n New active orbitals:\n")

for i in range(20, 40):
    ivs = np.argsort(mc.mo_coeff[:, i]**2)
    ss = ""
    for iv in ivs[::-1][:3]:
        ss += "%10.6f (%s)" % (mc.mo_coeff[iv, i] ** 2, ao_lb[iv].strip())
    print(" MO %2d :: %s" % (i, ss))

states = [(7, 2)]
ci = [
    (*mc.nelecas[::-1], 1, -1, mc.e_states[0], mc.ci[0].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[1], mc.ci[1].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[2], mc.ci[2].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[3], mc.ci[3].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[4], mc.ci[4].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[5], mc.ci[5].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[6], mc.ci[6].T),
    (*mc.nelecas, 1, 1, mc.e_states[0], mc.ci[0]),
    (*mc.nelecas, 1, 1, mc.e_states[1], mc.ci[1]),
    (*mc.nelecas, 1, 1, mc.e_states[2], mc.ci[2]),
    (*mc.nelecas, 1, 1, mc.e_states[3], mc.ci[3]),
    (*mc.nelecas, 1, 1, mc.e_states[4], mc.ci[4]),
    (*mc.nelecas, 1, 1, mc.e_states[5], mc.ci[5]),
    (*mc.nelecas, 1, 1, mc.e_states[6], mc.ci[6]),
]

siso = FCISISO(mol, mc, states)
siso.ci = ci
siso.ncore = 27
siso.norb = 7
energies = siso.kernel(dmao=dmao, amfi=True)

print(energies)

e0 = np.average(energies[0:6])
e1 = np.average(energies[6:14])

au2ev = 27.21139
print("")
print("E 2F(5/2)         = %10.4f eV" % (e0 * au2ev))
print("E 2F(7/2)         = %10.4f eV" % (e1 * au2ev))
print("2F(7/2) - 2F(5/2) = %10.4f eV" % ((e1 - e0) * au2ev))
print("2F(7/2) - 2F(5/2) = %10.4f cm-1" % ((e1 - e0) * au2ev * 8065.56))
