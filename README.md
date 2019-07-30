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

## Real-world CUDA

- Deep Learning is not O(lg N) on a GPU, it's 20-30x improvement
- O(1) is not O(1)
    - `a[i] += b[i]` 
    - `int c = 3 + 5`
- Memory access latency is a big thing (even for CPU implementations). 
    - Commonly the biggest cause of a slowdown, GPUs are rarely compute bound.

## Real-world CUDA

- grids of threads of blocks (insert image)
- While we can have as many threads as we want, they don't execute simultaneously
    - CUDA splits threads across blocks, and a batch of 32 threads (called a warp) 
    are executed simultaneously
    - GPUs have a Streaming multiprocessor which is assigned blocks and then runs the warps
    - RTX 2080Ti's have 68 SMs and 64 CUDA cores per SM.


## Tree Refressher

- Classification Trees is (are?) a greedy algorithm. 
- Given the various parameters/predictors, we scan across them to find the optimal split. We take the optimal split and then run the same algorithm on the splits.
- Two common extensions
    - Boosting (XGBoost) - We build an ensemble of trees and we reweight observations with error to improve performance. This tends to result in shallow trees
    - Random Forests - Build many trees on subsets of the data and predictors. Trees tend to be much deeper.

## Trees in CUDA

- Classification Trees
    - Depth First - each CUDA thread/block takes ownership

### Problems
    
- Performance is only a slight improvemnt for xgboost (3-10x) 
- 


## CUDA in non parallel

- EM - GMM equations. 
- No clear parallelisation. However, this is something that can benefit from the evaluation of above
- 


## Problems with GPUs

1. RAM
    - GPUs don't really have much more than 12 GB of RAM. System RAM can be huge, 128 GB
    - Deep Learning, SGD and batching works well. Less so with other approaches
2. Multi GPU
    - Non trivial algorithms required - magnitude of difficulty increase
    - It's significantly easier to run multiple single GPU experiments
    - Performance scaling can be poor, even within a single machine
3. Sequential Algorithms
    - XGBoost benchmarks show only a 4x speedup
    - MCMC doesn't benefit from CUDA cores
    - EM iterations still need to be done in sequence
4. Algorithm Changes
    - Algorithms such as DBScan rely on special data structures (R* trees) which required completely new ways of implementing this algorithm


5. Technical Debt
    - Experi
    - CUDA is not easy, sort of rare skill. 
    - Require 
6. No real ability to stream
    - RL shows this - data is collected in a simulated environment, batched, sent to GPU to train neural nets
    - Latency
