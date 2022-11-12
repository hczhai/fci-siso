
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf
from fcisiso import FCISISO
import numpy as np

# water (sto-3g)
mol = gto.M(atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """, basis='sto-3g', verbose=3, charge=1, spin=1, symmetry=True)
mf = scf.RHF(mol)
mf.kernel()

# atomic mean-field spin-orbit integral (AMFI)
hsoao = np.zeros((3, 7, 7), dtype=complex)
v = 2.1281747964476273E-004j * 2
hsoao[:] = 0
hsoao[0, 4, 3] = hsoao[1, 2, 4] = hsoao[2, 3, 2] = v
hsoao[0, 3, 4] = hsoao[1, 4, 2] = hsoao[2, 2, 3] = -v

# (nroots, multiplicity, wfnsym)
states = [(4, 2, 'A1'), (4, 4, 'B1'), (4, 4, 'A2'), (4, 4, 'B2')]
siso = FCISISO(mol, mf, states)
siso.kernel(hsoao=hsoao)

ref = np.array([
    -74.4304288802, -74.4304288802, -73.9779373072, -73.9779373072, -73.9779323581,
    -73.9779323581, -73.9733517557, -73.9733517557, -73.9733470568, -73.9733470568
])

print(np.linalg.norm(ref - siso.energies[:10]))
