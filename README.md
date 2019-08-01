# Understanding GPUs

With torch and tensorflow we have begun to rely on GPUs to speed up computations. Deep Learning or not, GPUs can provide massive computation speed ups but it’s not a panacea as NVIDIA would have you believe. Understanding how GPUs work can tell us where we should and shouldn’t use them.

## Abstract

Today, no one would even consider training a Neural Net without a GPU. Modern day Machine Learning techniques are heavily dependent on efficient computations of Linear Algebra operations - mechanically speaking, Deep Learning is simply matrix multiplications with non linear transforms - and these are the kind of operations that GPUs are incredibly efficient at doing.

This talk gives us the why of why GPUs are so good at these operations, and what else they might be good for. We give a high level overview of how GPUs work, with a dive into CUDA, it’s computation model (grids of blocks of threads) and it’s syntax. We learn to “Think with CUDA”, using this model to understand why matrix multiplication is so quick on a GPU and why Deep Learning get’s such a boost. We use our newfound knowledge to see how a decision tree (which has no linear algebra) can be sped up by a GPU.

And finally, we look at some limitations - the limits of multi GPU training, the large latencies involved in CPU - GPU communications resulting in some sunk cost, heavily sequential algorithms getting small improvements (Boosting), or algorithms that depend on efficient data structures needing new methods to work on a GPU (DBScan)


## Suggested Further Reading

1. https://github.com/nayyarv/PyCudaIntro - A companion CUDA piece with more code
2. Everything by Mark Harris