
"""
Ru 4+ ion in octahedral crystal field CAS(4e, 5o)
"""

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..")]

from pyscf import gto, scf, mcscf, fci
from pyscf.qmmm import mm_charge
from fcisiso import FCISISO, extract_ci_list
import numpy as np

mol = gto.M(atom='Ru 0 0 0',
            symmetry=False,
            basis='ano@6s5p3d',
            spin=0, charge=4, verbose=4)
dist = 1.5
chrg = -2.0
p_xyz = [[-dist, 0.0, 0.0], [dist, 0.0, 0.0],          
         [0.0, -dist, 0.0], [0.0, dist, 0.0],
         [0.0, 0.0, -dist], [0.0, 0.0, dist]]
p_chr = [chrg]*len(p_xyz)

mf = scf.newton(mm_charge(scf.RHF(mol), p_xyz, p_chr).sfx2c1e()).run(conv_tol=1e-10)

(ncaselec, ncas) = (4, 5)
ncore = (mol.nelectron - ncaselec) // 2
ci_tol = 1e-14
mySpins = [2, 0, 4]
myStates = [3, 6, 2]
nstates = sum(myStates)
weights = [1/nstates]*nstates

solver_list = []
states_list = []
for spin, states in zip(mySpins, myStates):
    solver = fci.direct_spin1.FCI(mol)
    solver.spin = spin
    solver.nroots = states
    solver.conv_tol = ci_tol
    s_square = (solver.spin/2)*(solver.spin/2 + 1)
    solver = fci.addons.fix_spin_(solver, ss=s_square, shift=1.5)
    solver_list.append(solver)
    states_list.append((states, spin+1))

print('ncore = ', ncore, ' ncas = ', ncas)

mc = mcscf.CASCI(mf, ncas, ncaselec)
mcscf.state_average_mix_(mc, solver_list, weights)
mc.kernel()

dmao = mc.make_rdm1()
ci_list = extract_ci_list(mc)

siso = FCISISO(mol, mc)
siso.ci = ci_list
siso.ncore = ncore
siso.norb = ncas
energies = siso.kernel(dmao=dmao, amfi=True)

print(energies)
print("For time being only one Ms component per state is interacting")
