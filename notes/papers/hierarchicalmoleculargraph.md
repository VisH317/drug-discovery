**Paper Title:** Hierarchical Molecular Graph Self-Supervised Learning for property prediction

**Paper Date:** Feb 2023

**Paper link:** [Article](https://www.nature.com/articles/s42004-023-00825-5)


**Intro/Current State**
- Current develpoments - applying language models to string-based molecular representations
  - RNNs, GRUs, LSTMs, Transformers applied to generate the strings
- self supervised learning methods => masked smiles modeling, string reconstruction
  - PROBLEM: strings 1 dimensional, no topological analysis
- Other current: Learning molecular motifs
  - substructures that are important for understanding
  - Pre-training tasks based on predicting motifs in an order
  - Other self-supervised learning tasks => molecular augmentation and contrastive learning (Wang et. al.)

**Challenges still present within these**
- How to capture molecular structure adequately
  - many use graph augment to contrast different views,
    - using edge modification, graph diffusion
  - still destroy structure and obscure molecule attributes
  - hard to preserve molecular structure and incorporate motifs
  - need to fuse more comprehensive information into molecular grpah representations
  - GNN or MPNN => current encoder backbone for molecules
- Fuse more comprehensive into molecular graph reprsentations
  - lacking global scale
- How to design pretraining tasks of self-supervised pre-training

**Their Solution**
- HiMol - Hierarchical molecular grpah self-supervied learning
- two major components
  - graph neural network - molecular encoder that addresses first two challenges
  - Combines nearby nodes into motifs => used to create the encoded graph representation
    - molecular motifs hold important property details (carboxyl => acidity)
    - motif decomposition rules => added together based on chemical criteria to preserve chemical characteristics
- MSP - multi-level self-supervised pre-training - 3 tasks
  - predict bond links, atom types, and bond types with atom representations