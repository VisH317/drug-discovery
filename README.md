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
- *Current* - getting the data cleaning to function
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