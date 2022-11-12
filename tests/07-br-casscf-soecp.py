
"""
Br atom CAS(7e, 4o)
"""

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf
from fcisiso import FCISISO
import numpy as np

ecp = """
Br nelec 10
Br ul
2	1.000000	0.000000
Br S
2	70.024257	49.962834
2	31.178412	370.014205
2	7.156593	10.241439
Br P
2	46.773471	99.112244	-198.224488
2	46.184120	198.253046	198.253046
2	21.713858	28.261740	-56.523480
2	20.941792	56.623366	56.623366
Br D
2	50.698839	-18.605853	18.605853
2	50.644764	-27.923280	-18.615520
2	15.447509	-0.379693	0.379693
2	15.500259	-0.780583	-0.520389
2	2.800391	0.035968	-0.035968
2	1.077480	0.094397	0.062931
Br F
2	14.465606	-1.091269	0.727513
2	21.234065	-2.887691	-1.443846
"""

mol = gto.M(atom='Br 0 0 0',
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
print("ZFS 2P  1/2 - 3/2 exp. = %10.4f cm-1" % 3685.24)

# https://physics.nist.gov/PhysRefData/Handbook/Tables/brominetable5.htm
# expr        3685.24 cm-1
# ccpv5z-pp   3688.7419 cm-1
# ccpvqz-pp   3687.8374 cm-1
# ccpvtz-pp   3680.1769 cm-1
# ccpvdz-pp   3671.9050 cm-1