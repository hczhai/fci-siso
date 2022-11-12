
"""
C atom CAS(4e, 4o)
"""

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf, fci
from fcisiso import FCISISO
import numpy as np

mol = gto.M(atom='C 0 0 0',
            symmetry=False,
            basis='ccpvtz-dk', spin=0, charge=0, verbose=3)
mf = scf.newton(scf.RHF(mol).sfx2c1e()).run(conv_tol=1e-14)

ncaselec = 4
ncore = (mol.nelectron - ncaselec) // 2
ncas = 4

print('ncore = ', ncore, ' ncas = ', ncas)

weights = np.ones(11) / 11
solver1 = fci.direct_spin1.FCI(mol)
solver1.spin = 2
solver1 = fci.addons.fix_spin(solver1, shift=0.2, ss=2)
solver1.nroots = 3
solver2 = fci.FCI(mol)
solver2.spin = 0
solver2.nroots = 3 + 5

mc = mcscf.CASSCF(mf, ncas, ncaselec)
mcscf.state_average_mix_(mc, [solver1, solver2], weights)

mc.kernel()

print(mc.nelecas, mc.e_states)

dmao = mc.make_rdm1()
print(dmao.shape)

states = [(3, 3), (5, 1)]
ci = [
    ((ncaselec - solver1.spin) // 2, (ncaselec + solver1.spin) // 2, 2, -2, mc.e_states[0], mc.ci[0].T),
    ((ncaselec - solver1.spin) // 2, (ncaselec + solver1.spin) // 2, 2, -2, mc.e_states[1], mc.ci[1].T),
    ((ncaselec - solver1.spin) // 2, (ncaselec + solver1.spin) // 2, 2, -2, mc.e_states[2], mc.ci[2].T),
    (*mc.nelecas, 2, 0, mc.e_states[3], mc.ci[3]),
    (*mc.nelecas, 2, 0, mc.e_states[4], mc.ci[4]),
    (*mc.nelecas, 2, 0, mc.e_states[5], mc.ci[5]),
    ((ncaselec + solver1.spin) // 2, (ncaselec - solver1.spin) // 2, 2, 2, mc.e_states[0], mc.ci[0]),
    ((ncaselec + solver1.spin) // 2, (ncaselec - solver1.spin) // 2, 2, 2, mc.e_states[1], mc.ci[1]),
    ((ncaselec + solver1.spin) // 2, (ncaselec - solver1.spin) // 2, 2, 2, mc.e_states[2], mc.ci[2]),
    (*mc.nelecas, 0, 0, mc.e_states[6], mc.ci[6]),
    (*mc.nelecas, 0, 0, mc.e_states[7], mc.ci[7]),
    (*mc.nelecas, 0, 0, mc.e_states[8], mc.ci[8]),
    (*mc.nelecas, 0, 0, mc.e_states[9], mc.ci[9]),
    (*mc.nelecas, 0, 0, mc.e_states[10], mc.ci[10]),
]

siso = FCISISO(mol, mc, states)
siso.ci = ci
siso.ncore = ncore
siso.norb = ncas
energies = siso.kernel(dmao=dmao, amfi=True)

print(energies)

e0 = np.average(energies[0:1])
e1 = np.average(energies[1:4])
e2 = np.average(energies[4:9])
e3 = np.average(energies[9:14])

au2cm = 219474.63
print("E 3P  (J = 1) - (J = 0) = %10.4f cm-1    exp. = %10.4f cm-1" % ((e1 - e0) * au2cm, 16.41671))
print("E 3P  (J = 2) - (J = 0) = %10.4f cm-1    exp. = %10.4f cm-1" % ((e2 - e0) * au2cm, 43.41350))
print("E 1D(J = 2) - 3P(J = 0) = %10.4f cm-1    exp. = %10.4f cm-1" % ((e3 - e0) * au2cm, 10192.66))

# https://physics.nist.gov/PhysRefData/Handbook/Tables/carbontable5.htm
# expr        16.41671   43.41350   10192.66   cm-1
# ccpv5z-dk   14.9004    44.6664    12747.9238 cm-1
# ccpvqz-dk   14.7194    44.1240    12754.9303 cm-1
# ccpvtz-dk   14.4282    43.2520    12772.6933 cm-1
# ccpvdz-dk   13.7174    41.1229    12838.2759 cm-1
