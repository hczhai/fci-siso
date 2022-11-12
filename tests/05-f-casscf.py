
"""
F atom CAS(7e, 4o)
"""

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf
from fcisiso import FCISISO
import numpy as np

mol = gto.M(atom='F 0 0 0',
            symmetry=False,
            basis='ccpvtz-dk', spin=1, charge=0, verbose=3)
mf = scf.newton(scf.RHF(mol).sfx2c1e()).run(conv_tol=1e-14)

ncaselec = 7
ncore = (mol.nelectron - ncaselec) // 2
ncas = 4

print('ncore = ', ncore, ' ncas = ', ncas)

mc = mcscf.CASSCF(mf, ncas, ncaselec).state_average_(np.ones(3) / 3.0)
mc.kernel()

print(mc.nelecas, mc.e_states)

dmao = mc.make_rdm1()
print(dmao.shape)

states = [(3, 2)]
ci = [
    (*mc.nelecas[::-1], 1, -1, mc.e_states[0], mc.ci[0].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[1], mc.ci[1].T),
    (*mc.nelecas[::-1], 1, -1, mc.e_states[2], mc.ci[2].T),
    (*mc.nelecas, 1, 1, mc.e_states[0], mc.ci[0]),
    (*mc.nelecas, 1, 1, mc.e_states[1], mc.ci[1]),
    (*mc.nelecas, 1, 1, mc.e_states[2], mc.ci[2]),
]

siso = FCISISO(mol, mc, states)
siso.ci = ci
siso.ncore = ncore
siso.norb = ncas
energies = siso.kernel(dmao=dmao, amfi=True)

print(energies)

e0 = np.average(energies[0:4])
e1 = np.average(energies[4:6])

au2cm = 219474.63
print("")
print("ZFS 2P  1/2 - 3/2      = %10.4f cm-1" % ((e1 - e0) * au2cm))
print("ZFS 2P  1/2 - 3/2 exp. = %10.4f cm-1" % 404.141)

# https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm
# expr        404.141 cm-1
# ccpv5z-dk   405.3439 cm-1
# ccpvqz-dk   400.8814 cm-1
# ccpvtz-dk   393.6671 cm-1
# ccpvdz-dk   376.8110 cm-1