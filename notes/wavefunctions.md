**Basic forms: 1 spinless particle in 1 dimension**
- $\Psi(x, t)$
  - IMPORTANT: $|\Psi(x,t)|^2$ gives the probability of a certain position of the particle at a certain time
- Bra-ket notation - linera algebra notation for linear operators in complex spaces
  - ket: $\ket{v}$ - denotes a vector and represents state of a quantum system
  - bra: $\bra{f}$ - linear map that maps each vector in vector space $V \rArr C$
  - when combined => show full linear map with specifics 
  - Quantum state - mathematical representation of a quantum system
    - usually represented as vectors or kets


**Molecular Orbitals**
- arise from interactions between atomic orbitals => based on group theory
  - based on constructive and destructive interference of atomic orbitals and how close in energy each orbital is
  
**Qualitative Concepts in Molecular Orbitals**
- Linear combinations of atomic orbitals - approximation of molecular orbital as a linear combination of each atom's orbital
  - for diatomic molecule: $\Psi = c_a\psi_a + c_b\psi_b$
- Atomic orbital interactions:
  - **Bonding MOs** - bonding interactions between atomic orbitals are constructive interactions
    - the total MO has lower energy than the atomic orbitals that combine to make them
  - **Antibonding MOs** - interaction between atomic orbitals with destructive interactions
    - higher in energy than the atomic orbitals
  - **Nonbonding MOs** - no interaction, same energy as atomic orbitals in the molecule
- Interaction can also be categorized by symmetry of molecular orbitals
  - each orbital type (s, p, d, f, g) have different symmetry
  - $\sigma$ symmetry - MO with this results from interaction of two atomic s orbitals or two atomic $p_z$ orbitals
  - $\pi$ symmetry - results from interaction of either $p_x$ or $p_y$ orbitals => produces a phase change when assymetric
  - $\delta$ symmetry - results from interactions of two atomic $d_{xy}$ or $d_{x^2-y^2}$ orbitals, usually in transition metal complexes
  - $\phi$ summetry - higher order bonds overlapping
- Molecules with centers of invesrion (centrosymmetric) have other symmetry labels
  - centrosymmetric:
    - homonuclear molecule - only composed of one element
    - octahedral - molecule with 6 ligands arranged around a central atom
    - square planar - 4 molecules arranged in square around central atom
  - non-centrosymmetric: 
    - heteronuclear - different elements in molecule
    - tetrahedral - 4 different element molecules around the central atom => leads to differing bond angles

**Bonding in molecular obitals**
- orbital degeneracy - molecular orbitals are degenerate when they have the same energy
  - example: homonuclear diatomic molecules have molecular orbitals derived from $p_x$ and $p_y$ orbitals
  - when non-degenerate - in the ground state
  - degeneracy - number of different possible states in an energy level


**Computational Derivation**
- Use a basis set - contains the list of all atomic orbitals, so that each vector can represent a linear combination of them easily
  - contain gaussian functions of different orbitals