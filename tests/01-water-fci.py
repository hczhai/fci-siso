
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

try:
    import block2
except:
    pass

from pyscf import gto, scf
from fcisiso import FCISISO
import numpy as np

# water (sto-3g)
mol = gto.M(atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """, basis='sto-3g', verbose=3, symmetry=True)
mf = scf.RHF(mol)
mf.kernel()

# atomic mean-field spin-orbit integral (AMFI)
hsoao = np.zeros((3, 7, 7), dtype=complex)
v = 2.1281747964476273E-004j * 2
hsoao[:] = 0
hsoao[0, 4, 3] = hsoao[1, 2, 4] = hsoao[2, 3, 2] = v
hsoao[0, 3, 4] = hsoao[1, 4, 2] = hsoao[2, 2, 3] = -v

# (nroots, multiplicity, wfnsym)
states = [(4, 1, 'A1'), (4, 3, 'B1'), (4, 3, 'A2'), (4, 3, 'B2')]
siso = FCISISO(mol, mf, states)
siso.kernel(hsoao=hsoao)

ref = np.array([
    -74.9319193494, -74.5130378800, -74.5130377093, -74.5130376898, -74.3781961604,
    -74.3781959410, -74.3781957210, -74.3350363387, -74.3350359320, -74.3350358335
])

print(np.linalg.norm(ref - siso.energies[:10]))
