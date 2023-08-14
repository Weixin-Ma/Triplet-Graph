# Triplet-Graph
The official implementation of Triplet-Graph.

The code will be open-sourced upon acceptance of the paper.

<p align="center"><img src="demo.gif" width=800></p>

# Something about Triplet-Graph
## Semantic Graph 
A semantic graph is an abstract representation of an input 3D point cloud. Each vertex in the graph refers to the centroid of an object. Two vertices will be connected when their distance is less than a pre-defined threshold.
<p align="center"><img src="graph.gif" width=800></p>

## Triplet
The triplet is the basic element that we use to extract histogram-based descriptor for both individual vertex (local descriptor) and the whole graph (global descriptor). A triplet is a group of three connected vertices in the graph. In a semantic graph, all the triplets that use a certain vertex as the middle vertex are used to extract the local descriptor.
<p align="center"><img src="triplet.gif" width=800></p>

## Local Descriptor
<p align="center"><img src="descriptor.png" width=450></p>

# [Support Material](appendix.pdf)
Here are the support material, which investigates the effects of some important parameters in Triplet-Graph. 
