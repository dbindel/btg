# BTG: Fast Bayesian Transformed Gaussian (BTG) modeling

The HSS fast matrix solver can be deconstructed into three parts: convert a SPD matrix into HSS representation, factorize the HSS representation, and finally do forward and backward substitution.

For converting SPD matrix into HSS representation, the basic routine is to follow algorithm  3.1. As said, the number of blocks should be 2^p (with dimension n*n). This step should cost O(N^2) run time. At this point, I have implemented p = 4, following the example given in Section 3. Theoretically understand how the algorithm works, still thinking about a good way to code up.

For factorizing HSS representation, refer to algorithm 4.1 (superfast HSS Cholesky). Two helper functions are used to help find out the child of the postorder traversal of tree, and also check whether a node is leaf node (i.e findchild and ischildnode). A few points that I’m a little bit confused:
 	- The orthogonal matrix Q_i is obtained from QL factorization with respect to U_i. 
	- Referring to the example in Section 4.2.4, there is a permutation matrix P_i applied to the factorization. 

For forward and backward substitution, TBC.
