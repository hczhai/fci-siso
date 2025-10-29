import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf
from fcisiso import FCISISO
import numpy as np

mol = gto.M(atom='Cr 0 0 0',
            symmetry=False,
            basis='ccpvtz-dk', spin=1, charge=5, verbose=3)
mf = scf.newton(scf.RHF(mol).sfx2c1e())
mf.kernel()

print("\n Molecular orbital components:\n")

# total: 19 elec
# core: 1s2 2s2 2p6 3s2 3p6 (18e, 9o)
# active: 3d1 (1e, 5o)
ao_lb = gto.sph_labels(mol)
for i in range(40):
    ivs = np.argsort(mf.mo_coeff[:, i]**2)
    ss = ""
    for iv in ivs[::-1][:3]:
        ss += "%10.6f (%s)" % (mf.mo_coeff[iv, i] ** 2, ao_lb[iv].strip())
    print(" MO %2d :: %s" % (i, ss))

print("\n CASSCF(1, 5):\n")

cas_list = [10, 11, 12, 13, 14]
mc = mcscf.CASSCF(mf, 5, 1)
mc.fix_spin_(ss=0.75)
mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list, base=0)
mc.state_average_((0.2, 0.2, 0.2, 0.2, 0.2))
mc.kernel(mo)

print(mc.nelecas, mc.e_states)

dmao = mc.make_rdm1()
print(dmao.shape)

print("\n New active orbitals:\n")

for i in range(10, 30):
    ivs = np.argsort(mc.mo_coeff[:, i]**2)
    ss = ""
    for iv in ivs[::-1][:3]:
        ss += "%10.6f (%s)" % (mc.mo_coeff[iv, i] ** 2, ao_lb[iv].strip())
    print(" MO %2d :: %s" % (i, ss))

states = [(5, 2)]
ci = [
    (*mc.nelecas[::-1], 1, -1, mc.e_states[0], mc.ci[0].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[1], mc.ci[1].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[2], mc.ci[2].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[3], mc.ci[3].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[4], mc.ci[4].T),
    (*mc.nelecas, 1, 1, mc.e_states[0], mc.ci[0]),
    (*mc.nelecas, 1, 1, mc.e_states[1], mc.ci[1]),
    (*mc.nelecas, 1, 1, mc.e_states[2], mc.ci[2]),
    (*mc.nelecas, 1, 1, mc.e_states[3], mc.ci[3]),
    (*mc.nelecas, 1, 1, mc.e_states[4], mc.ci[4]),
]

siso = FCISISO(mol, mc, states)
siso.ci = ci
siso.ncore = 9
siso.norb = 5
energies = siso.kernel(dmao=dmao, amfi=True)

print(energies)

e0 = np.average(energies[0:4])
e1 = np.average(energies[4:10])

au2ev = 27.21139
print("")
print("E 2D(3/2)         = %10.4f eV" % (e0 * au2ev))
print("E 2D(5/2)         = %10.4f eV" % (e1 * au2ev))
print("2D(5/2) - 2D(3/2) = %10.4f eV" % ((e1 - e0) * au2ev))
print("2D(5/2) - 2D(3/2) = %10.4f cm-1" % ((e1 - e0) * au2ev * 8065.56))
