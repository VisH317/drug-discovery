## The cluster algorithm

_from following source:_ [hgraph2graph](https://github.com/wengong-jin/hgraph2graph/blob/master/hgraph/mol_graph.py)

**Finding clusters:**
- Get the number of atoms: if its 1 - return a single cluster
- iterate through all bonds
  - if the bond isn't in a ring it is a bridge bond - add it to the list of clusters
- Use Chem.GetSymmSSSR to get the non-overlapping set of rings
- make sure that the first item in the clusters list has the root element (0 index)
- create the atom_clusters dict: maps an atom to the clusters that its a part of

**Tree Decomposition Algorithm**
- iterate through each atom and cluster pair in atom_cls
- initialize empty graph with nodes for len(clusters)
- for each: 
  - if no clusters - contniue
  - make a list of all the bridge bonds its a part of, and same for rings (by checking cluster sizes)
  - if there are at least 2 bonds and the cluster isn't in a ring - it can be considered a cluster by itself
    - draw edge to the nearby clusters, weighted 100
  - if the atom is part of more than 2 rings it should be considered its own cluster
    - connected to each ring cluster, weighted 100
  - otherwise: loop through relationship between every cluster on the specific atom
    - add a weight value based on how related the two clusters are to create the final graph
