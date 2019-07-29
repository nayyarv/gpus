# Understanding GPUs

With torch and tensorflow we have begun to rely on GPUs to speed up computations. Deep Learning or not, GPUs can provide massive computation speed ups but it’s not a panacea as NVIDIA would have you believe. Understanding how GPUs work can tell us where we should and shouldn’t use them.

## Abstract

Today, no one would even consider training a Neural Net without a GPU. Modern day Machine Learning techniques are heavily dependent on efficient computations of Linear Algebra operations - mechanically speaking, Deep Learning is simply matrix multiplications with non linear transforms - and these are the kind of operations that GPUs are incredibly efficient at doing.

This talk gives us the why of why GPUs are so good at these operations, and what else they might be good for. We give a high level overview of how GPUs work, with a dive into CUDA, it’s computation model (grids of blocks of threads) and it’s syntax. We learn to “Think with CUDA”, using this model to understand why matrix multiplication is so quick on a GPU and why Deep Learning get’s such a boost. We use our newfound knowledge to see how a decision tree (which has no linear algebra) can be sped up by a GPU.

And finally, we look at some limitations - the limits of multi GPU training, the large latencies involved in CPU - GPU communications resulting in some sunk cost, heavily sequential algorithms getting small improvements (Boosting), or algorithms that depend on efficient data structures needing new methods to work on a GPU (DBScan)

# Talk Skeleton

## TOCs

- This is a high level view of GPUs and CUDA from a math-y perspective
- Any detailed questions re 

## Why GPUs?

What unites

- Video Games?
- Crypto-currency mining?
- Machine Learning (especially the Deep kind)?

## Crypto

- Blockchains work by requiring nodes to solve a hard problem by random search.
- Blockchain is fundamentally a distributed algorithm (across nodes, but should scale within nodes too)

## Video Games

- 3d graphics are done by defining polygons and movement is achieved by rotations, skews and translations
- Matrix operations underpin graphics processing

## Machine Learning

- Deep Learning is fundamentally matrix operations 
- Fully Connected Layer == Wx + b == Matrix multiply and translate
- Many Machine Learning algorithms are fundamentally linear algebra operations and others are naturally parallelisable

## Why GPUs (conc.)

- Very good at parallelisable tasks
- Linear Algebra tends to be very parallelisable, hence ML gets the speed up. 


## GPU vs CPU

CPUs

- High single threaded performance (5.0 GHz), minimal core capacity (8-16)
- Large memory amounts2
- Limited control of caches, so performance can be very variable
- Aimed at being general purpose

GPUs

- Multiple cores with shared memory pool
- Memory tends to be limited (max 12 GB today)
- Full control of memory means code is easier to optimize (`__shared__`)
- Very Specific

## What is CUDA?

- Stood for Compute Unified Device Architecture but no one cared so this was forgotten
- CUDA devices have Streaming Multiprocessors (SMs) and each has a number of CUDA cores. GTX970 has 13 SMs with 128 CUDA cores each.
- In CUDA, you need plenty of independent threads to take advantage of the architecture and to minimize memory latency via async scheduling.
- Not many guarantees of all threads running exactly in parallel, so code still needs to be thread safe.
- number of threads is magnitudes greater than in standard multicore programming.

## Thinking with CUDA

- Theoretical CUDA concept
    - Infinite threads running the same time
    - Instant Memory Access
- These are both lies, but a good starting point

## Summing two Arrays

- `a` and `b` of length N
- Single Core = O(N)
- CUDA
    - N threads - `a[i] += b[i]`
    - O(1)

## Sum an Array

- `a` is of length N
- Single Core = O(N)
- CUDA
    - N/2 threads
    - Add first half of `a` to second half.
    - Add first quarter of `a` to second quarter
    - ...
    - O(lg N)
- *Halving gets inefficient, so eventually a single thread takes over*

## Matrix Multiply

- `a @ b`
- N^2 output elements, of which each is a dot product
- Single Core = O(N^3) or O(N^2.81)
- CUDA
    - Spawn N^3 threads
    - Each N^2 output element has N threads dedicated calculcating the dot product
    - Dot product - multiply two elements and sum = O(1) + O(lg N) = O(lg N)
    - Hence Matmul is O(lg N)


