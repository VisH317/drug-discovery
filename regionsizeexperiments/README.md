# Region Size Experiments

_The point of this folder:_ To determine which region sizes have the most effect when predicting molecular properties

**Ideas:**
 - different local regions of initial processing can determine different features in the molecule
 - When looking at PAMnet, the local features (one-two in region) had high effect, while global parameters rarely had a significant effect (only in one benchmark)
 - Which level of regions can be related to each other


**Controls:**
 - number of encoders
 - types of attention
 - number of attention heads
 - topological attention augmentation