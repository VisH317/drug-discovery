**Points of improvement with PAMNet:**
 - Cannot consider all relationships - only close local + far 
 - performs best on QM9 - small molecules with less than 9 non-hydrogen atoms per item
   - need to be able to generalize to larger molecules more efficiently
   - might be because not starting with small enough regions, local to global is too big of a jump which makes global kinda useless, mainly due to bias in training data
 - NEED TO ADD THE GREATEST POSSIBLE PRETRAINED INDUCTIVE BIAS

**Main ideas to optimize for**
 - bigger molecules with high accuracy
 - FASTER SPEED
 - generalization - the foundational model of molecular properties
 - simulations?
 - contrastive learning task?

**Embeddings + Preprocessing + Pos enc**
 - Similar to vision transformers - chunk it up into arbitrary convolution-based items as groups (not full molecules, but very small regions at first)
 - preprocessing steps: pooling each chunk based on bonds and quantum properties using a convolution operation
 - positional encoding - relative to other regions

**Features to use**
 - quantum number calculations
 - others: hybridization, bond relations

**Attention mechanism architecture**
 - Tiered group query attention
 - Multi query attention
 - different types of attention at different levels
   - first - within region attention
   - second - neighboring region attention
   - tiered in separate encoder layers vs in the same
 - 

**Overall architecture**
 - architecture 1:
   - chunk up arbitrarily, preprocess based on bonds and add relative positional encoding
   - run through three layers, first layer takes in local relationships, second medium, third far relationships (can configure multi tiered attention in each)
   - output trained on multiple tasks at once
 - architecture 2:
   - same as architecture one but multiple tiers in the same layer
 - architecture 3:
   - evolutionary simplification algorithm - same region based chunking
   - bonds generalized between regions at each step to simplify graph
   - eventually reaches embedding value
 - architecture 4:
   - fine-grained attention - each molecule chooses a candidate to combine into region iteratively
   - continues until no items left => leaves only the embedding
   - CAN ALSO GO SEQUENTIALLY - LIKE HOW PROTEINS ARE BUILT
 - architecture 5:
   - train a wave function approximator
   - use it to pass one by one messages between nodes
   - simple message passing, no global attention
 - architecture 6:
   - contrastive learner between two regions


**Summaries**
1. chunk up and have multiple encoder layers to simplify over itme
2. chunk up but have multiple tiers per encoder layer
3. determines relationships in iterations based on proximity at each time step until final embedding reached
4. Similar to 3 but based on sequential protein building
5. wave function approximator function: throw into GCN or attention network

**Things to test now**
1. Regional encoding + transformer with different types of attention
2. Adding a region combining algorithm
3. using tiered attention

**Atom Features**
 - Atomic number
 - hybridization
 - valence electrons

**Bond features for regions**
 - Bond length
 - bond order
 - magnetic data - nmr shielding
 - natural bond orbital analysis

**Global regional features**
 - polarization
 - electron density
 - molecular energy (hartree-fock)
 - regional chirality
 - torsional/dihedral angles - rotation around a bond axis
 - Resonance structures - ways to represent electron distribution in molecule using Lewis structures
 - steric hindrance - repulsion between atoms or groups in proximity - USEFUL FOR BETWEEN REGIONS as well

**Region Creation network**
 - Get nearest atoms direct bonds - embed each atom and each bond's quantum representation
 - Run a single internal convolution of message passing between the local area and convert the region using pooling with edges and information on other nodes, take the final middle node region as the embedding
   - FUTURE: add the angle information as well
 - end up with a bunch of atoms with regional encodings
 - PAMNet - did local and all over global
   - we can do this by throwing it into a transformer for better analaysis
   - or have controlled encoders/attentions that first apply to local regions to create individual output embeddings
     - This would work by selecting regions near each other at first, passing them through one attention pass for nearby
     - Take the output of this attention and use in future attentions
 - IMPORTANT: add psi4 molecule quantum information along the way at the end of each attention step
 - OR: instead of regions - can have different encoders dedicated to different regions, then encoder values from all are combined
 - OR: can have inner attention in ones, and then global cross attention in others
   - different paradigms to encode at each encoder layer
