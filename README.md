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

# Intro 

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
- Large memory amounts - 128GB ++ 
- Limited control of caches, so performance can be very variable
- Aimed at being general purpose

GPUs

- Multiple cores with shared memory pool
- Memory tends to be limited (max 12 GB today)
- Full control of memory means code is easier to optimize (`__shared__`)
- SIMD - Single Instruction, Multiple Data (locality)

## What is CUDA?

- Stood for Compute Unified Device Architecture but no one cared so this was forgotten
- CUDA devices have Streaming Multiprocessors (SMs) and each has a number of CUDA cores. GTX970 has 13 SMs with 128 CUDA cores each.
- In CUDA, you need plenty of independent threads to take advantage of the architecture and to minimize memory latency via async scheduling.
- Not many guarantees of all threads running exactly in parallel, so code still needs to be thread safe.
- number of threads is magnitudes greater than in standard multicore programming.

## Syntax and terminology

- CPU and System RAM = Host
- GPU RAM = device/global
- Code that runs (in parallel) on a GPU is called a kernel and defined with `__global__` keyword
- Cache control - `__shared__`, shared across a block
- `__syncthreads` is how we ensure synchronization. Some penalty, but less than you'd expect


# Thinking with CUDA

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

# Reality

## Real-world CUDA

- Deep Learning is not O(lg N) on a GPU, it's 20-30x improvement
- O(1) is not O(1)
    - `a[i] += b[i]` 
    - `int c = 3 + 5`
- Memory access latency is a big thing (even for CPU implementations). 
    - Commonly the biggest cause of a slowdown, GPUs are rarely compute bound.
    - Stuff like convolution layers are less memory bound 
- Memory Latency
    - Copying Data from RAM to GPU memory - slow, but with high bandwidth
    - Data from GPU RAM to SM (copy to SM cache if possible)
    - SM caches are 100x faster than GPU RAM, but can have memory bank conflicts

## Real-world CUDA

- grids of threads of blocks (insert image)
- While we can have as many threads as we want, they don't execute simultaneously
    - CUDA splits threads across blocks, and a batch of 32 threads (called a warp) 
    are executed simultaneously
    - GPUs have a Streaming multiprocessor which is assigned blocks and then runs the warps
    - RTX 2080Ti's have 68 SMs and 64 CUDA cores per SM.
- Blocks are assigned to the SMs
- NVIDIA provides spreadsheets on optimal block/thread combinations for your data


## Real World Matmul

Naive
- Read row and column and compute dot product
- Each element will need repeated access (N times per matrix) from GPU memory

Tiled
- Take submatrix tiles and multiply
- We then sum up for our submatrix stride

cuBLAS
- Very heavily optimised, memory latency is barely an issue

Convolutions
- Implemented as a Matmul, to take advantage of benefits
- Efficiency comes from the tiling being done on local caches


# CUDA in ML

## Tree Refresher

- Classification Trees is (are?) a greedy algorithm. 
- Given the various parameters/predictors, we scan across them to find the optimal split. We take the optimal split and then run the same algorithm on the splits.
- Two common extensions
    - Boosting (XGBoost) - We build an ensemble of trees and we reweight observations with error to improve performance. This tends to result in shallow trees
    - Random Forests - Build many trees on subsets of the data and predictors. Trees tend to be much deeper.

## Trees in CUDA

- Classification Trees
    - Depth First - each CUDA SM/block takes ownership of a single parameter and use the warp to choose optimal split. This provides maximum speedup at early splits.
    - Breadth First - each CUDA SM/block is tasked with generating the full tree. This is more effective when the initial splits have already happened. 

### Problems
    
- Performance is only a slight improvemnt for xgboost (3-10x) 
- Much greater for random forests (can construct multiple trees simultaneously)
- Speedup is greatest when the features are few. 


## CUDA where you wouldn't expect it

- EM - GMM equations. 
- No clear parallelisation. However, this is something that can benefit from the evaluation of above
- However, the two summations can be parallelised quite easily.
- The objective function can be sped up 25x 
- EM is still iterative (as is gradient descent) and 


## Problems with GPUs

1. RAM
    - GPUs don't really have much more than 12 GB of RAM. System RAM can be huge, 128 GB.
    - Deep Learning, SGD and batching works well. But maybe we want to increase batch size at later epochs?
        - Large models can have 65 million parameters = 1 Gb with Float16 rep. This is not an insignificant part of GPU RAM.
2. Multi GPU
    - Video gamers knew this years ago
    - Non trivial algorithms required - magnitude of difficulty increases
    - It's significantly easier to run multiple single GPU experiments
    - Performance scaling can be poor, even within a single machine. 
        - Convolutions scale well, others not so well (1 GPU may be fastest)
    - Many ML algorithms have identifiability which means independent lock free depends on sparsity assumptions that may not hold. (HOGWILD!) 
3. Sequential Algorithms
    - XGBoost benchmarks show only a 4x speedup
    - MCMC doesn't benefit from CUDA cores
    - EM iterations still need to be done in sequence
    - Time series 
    - RNNs training benefit is much reduced for this reason (Without NVIDIA provided code, these were slower than the CPU)
4. Algorithm Changes
    - Algorithms such as DBScan rely on special data structures (R* trees) which required completely new ways of implementing this algorithm
    - XGBoost (trees in general) and KNN require fundamental algorithmic changes to work correctly

5. Technical Debt
    - Compute power is not automatically available without the software. We're limited to what is available. Nvidia is good and proactive, but if you're off the beaten track, they're not very helpful
    - CUDA is non trivial to program in. Most researchers will need a large engineering team to get anything out of their GPUs if they'
    - Requires 

6. No real ability to stream (i.e. small amounts continuously)
    - Latency on input - consider running a neural network forward on visual or auditory data as it comes in. GPUs are not good due to the high latency on CPU - GPU memory latency
    - RL shows this - data is collected in a simulated environment, batched, sent to GPU to train neural nets. 
        - eg DQN batches with replay

# Conclusions

## My Best Practices with GPUs (Personal Opinion)

- Personal use
    - Your video game card
    - Colab
- Check what algorithms you're using before buying. There is more to ML than deep learning.
    - You might still save in power
    - You might still get a speedup despite not being obviously
- Device RAM is the commonly biggest limitation. It's cheaper to buy the one with the most RAM than it is to pay your ML Engineer/DS to fiddle with smaller GPUs

- Buying your own hardware tends to pay itself of quickly vs cloud offerings. Though there can be a lot of overhead in managing hardware (especially at mid scale)
    - Much easier to do experiments on local hardware since it avoids data transfer issues.
    - Using GPUs to deliver inference is always going to be cloud based
- Multi GPU
    - Multi-GPU setups are best utilised doing individual contained experiments on each GPU instead of multiple GPU training.
    - Training on many GPUs should wait till your team has a mature tech stack and a decent engineering team (lot's of debugging re utilization)
    - Heat is a non trivial issue. Homemade GPU boxes never take this into account. 
