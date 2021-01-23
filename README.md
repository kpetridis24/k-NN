# k-NN
C program to locate the k nearest neighbors using sequential, distributed-parallel algorithm.
For our project we use random metric spaces in the form of multidimensional matrices, my implementation has been tested on high-performance-computers with matrices of dimensions up to n=100.000 and d=500. n is number of rows, thus number of points, and d is the number of dimensions, thus the number of coordinates of each point.

~ Function V0 ~

The V0 function is the basis of our work since it implements the general logic of finding and storing k-NN.
After the calculation of each distance, we find the exact location to insert it (if it exists) in our “ndist” array
using binary search algorithm. The remaining elements starting from the previously mentioned location are
then shifted by one position to the right.

~ Function V1 ~

This function distributes the calculation of the k-NN to MPI processes, each of which holds a fixed part of
the initial array (local corpus points), as well as a part which is sent each time to the next and received from
the previous (incoming query points). The queries move in the “ring” of processes until all distances are
computed and the k-NN are detected. Its functionality is represented in the following figure.

~ Function V2 ~

In the last part of the exercise, the k-NNs are found in log(n) steps using an approximately binary VP tree.
The tree is created by a function which returns the struct pointer that represents its vertex. Recursively
dividing our space we end up in a subspace of size four (contains 4 nodes) so we consider the first one as a
leaf (isLeaf=1) and the remaining three as its children (kids). While creating the tree, we store the necessary
information at each node, thus the vantage point, median distance, left subtree, right subtree, parent node and
variables to indicate whether it is itself a left or right subtree, or even a leaf. This facilitates the downward
and the upward search on the VP tree. The “searchVPT” function accepts a query point p as argument and
searches its k-NN on a tree, selecting each time as subspace for searching the one where the point p is
located in. When the search is completed (detected all leafs and checked their children), we check if the
farthest neighbor in the list, intersects another subspace and if necessary, further searches are performed. The
need to send the VPT structure using openMPI prompted me to create serialization-deserialization functions,
which create a list containing all the information about each node that is required in order to rebuild it on the
receiving process.The recreation of the VP tree is accomplished by reading the list items and using the “newNode” function
which, accepting the above mentioned variables, generates all nodes recursively


~ Code Weaknesses ~

V0: none
V1: correct indices of k-NN (nidx array) but without offset
V2: some issue in "searchVPT()" causes few incorrect k-NN updates

