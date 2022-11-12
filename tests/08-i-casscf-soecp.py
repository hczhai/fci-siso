
"""
I atom CAS(7e, 4o)
"""

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf
from fcisiso import FCISISO
import numpy as np

ecp = """
I nelec 28
I ul
2       1.000000        0.000000
I S
2       40.033376       49.989649
2       17.300576       281.006556
2       8.851720        61.416739
I P
2       15.720141       67.416239       -134.832478
2       15.208222       134.807696      134.807696
2       8.294186        14.566548       -29.133096
2       7.753949        28.968422       28.968422
I D
2       13.817751       35.538756       -35.538756
2       13.587805       53.339759       35.559839
2       6.947630        9.716466        -9.716466
2       6.960099        14.977500       9.985000
I F
2       18.522950       -20.176618      13.451079
2       18.251035       -26.088077      -13.044039
2       7.557901        -0.220434       0.146956
2       7.597404        -0.221646       -0.110823
"""

mol = gto.M(atom='I 0 0 0',
            symmetry=False,
            basis='ccpvtz-pp', ecp=ecp,
            spin=1, charge=0, verbose=3)
mf = scf.newton(scf.RHF(mol)).run(conv_tol=1e-14)

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
print("ZFS 2P  1/2 - 3/2 exp. = %10.4f cm-1" % 7603.15)

# https://physics.nist.gov/PhysRefData/Handbook/Tables/iodinetable5.htm
# expr        7603.15 cm-1
# ccpv5z-pp   7337.2339 cm-1
# ccpvqz-pp   7338.1457 cm-1
# ccpvtz-pp   7324.7639 cm-1
# ccpvdz-pp   7325.8723 cm-1