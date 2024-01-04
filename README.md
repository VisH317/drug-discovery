# Main idea

_PAMNET:_ Current state of the art, does not use local information that much, and is only state of the art in qm9, which is known to have much smaller molecules than normal

**Our Goal:**
- Make a network that is able to generalize to much larger molecules over time
- Need to capture better dependencies => transformers with extensive pre training

**The main starting architecture:**
- Initial preprocessing - a GAT to take in local dependencies (need to configure the range of this)
- After - topological relative encoding with transformer - hope that attention will capture more robust global dependencies
  - proff - Elembert - preprocessing was pre-made cluster based instead of dynamic

**Idea Progression to Test:**
- *Current* - getting the data cleaning to function - DONE
- Getting the transformer to function
- Testing different local region size: one with just bond lengths, maybe adding bond angles?
- Adding in global dependencies - molecular energy, orbital energies, or estimating them ourselves?
- Splitting based on groups - chunk the graph like a vision transformer and run on individual chunks
  - need to decide on topological encoding
  - chunks increasing in size over time?
- Research molecular wavefunction and orbitals - how could it be estimated more accurately?
  - maybe optimize to outside parts on the molecule and their efficacies?
- Trying pretraining tasks?
  - predicting a part of the molecule?
  - predicting which items would bond


**The new idea**
- wavefunction data augmented attention
- Add info to Q + K matrices
  - radial function estimator and spherical harmonic estimator
  - adds along with globals => gives a better output?
- Plus local details

**My Main Idea**
- wavefunction estimation for substructures
- hierarchical attention

**Another Idea**
- estimating a LCAO
- generate coefficients based on wavefunction attention


**How to make wavefunction-based embeddings**
- first - generate radial and spherical basis sets
- for radial and spherical - generate coefficients for each basis
  - add them up => gives us the wavefunction approximation
- for substructures - generate coefficients for LCAO
- how to integrate the wavefunction or molecular wavefunction into attention
  - first comparison - basic info like atomic number, mass, hybridization
  - + topological - adds a basic positioning system to understand relationships
  - + radial wavefunction - adds atomic distance estimations
  - + spherical wavefunction - 
  - the wavefunction representation will be a list of coefficients
    - when attending to two different atoms, they 


**The final idea**
- two main aspects: substructures and wavefunction estimations
- the wavefunction estimation
  - coefficients of radial and spherical bases
  - will be LCAO-ed together for substructures
- substructure searching
  - one layer - will attend to atom-level relationships
  - other layers - detect repeating substructures just like JT-VAE
    - create a wavefunction-based embedding - LCAO estimation based on the coefficient estimation from each atom
      - done through local attention =>


**The actual final idea**
- Substructure
  - each substructure is detected
  - local attention is done within the substructure - spherical + radial bessel function analysis
  - we get a substructure embedding
- The ACTUAL THING
  - attention is done, but substructure embeddings are inserted in
  - also some feedback loop to update the substructure embeddings based on the attention value?
- Things
  - substructure detection
  - substructure analysis - message passing + LCAO-inspired aggregation or... a TRANSFORMER!
  - then: global - attach a substructure embedding during attention
    - used to compute relations
    - add the substructure embedding to each atom when relevant
      - have a vector v_nosub for when not in substructure
    - keep the positional attention embedding
      - need to fine tune this with some sort of positioning map
    - create the relations and compute attention
      - when atoms in same substructure - no effect
      - atoms in different substructure - add to the substructure embedding slightly
        - this will most likely be a simple linear layer that takes in distance + relative properties