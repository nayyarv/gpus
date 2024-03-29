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

## What is CUDA?

- Stood for Compute Unified Device Architecture but no one cared so this was forgotten
- CUDA devices have Streaming Multiprocessors (SMs) and each has a number of CUDA cores. 
    - GTX970 has 13 SMs with 128 CUDA cores each.
    - RTX 2080 Ti has 68 SMs with 64 CUDA cores each
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


## Real-world CUDA Threads

- grids of threads of blocks (insert image)
- While we can have as many threads as we want, they don't execute simultaneously
    - CUDA splits threads across blocks, and a batch of 32 threads (called a warp) 
    are executed simultaneously
    - GPUs have a Streaming multiprocessor which is assigned blocks and then runs the warps
    - RTX 2080Ti's have 68 SMs and 64 CUDA cores per SM.
- Blocks are assigned to the SMs which run warps in an async-esque approach. When threads pause for data access, they are shuffled off till the data makes it across
- NVIDIA provides spreadsheets on optimal block/thread combinations for your data


## Real-world CUDA Memory

- Deep Learning is not O(lg N) on a GPU, it's 20-30x improvement
- O(1) is not O(1)
    - `a[i] += b[i]` 
    - `int c = 3 + 5`
- Memory access latency is a big thing (even for CPU implementations). 
    - Commonly the biggest cause of a slowdown, GPUs are rarely compute bound.
    - Stuff like convolution layers are less memory bound 
- Memory Latency
    - Copying Data from RAM to GPU memory - slow, but with high bandwidth
    - Data should be accessed in region blocks for speed. 
    - Data from GPU RAM to SM (copy to SM cache if possible)
    - SM caches are 100x faster than GPU RAM, but can have memory bank conflicts

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
- Recursive algorithm - once split has happened, the algorithm can be rerun on the split data.
- Two common extensions
    - Boosting (XGBoost) - We build an ensemble of trees and we reweight observations with error to improve performance. This tends to result in shallow trees
    - Random Forests - Build many trees on subsets of the data and predictors. Trees tend to be much deeper in this setup. 

## Trees in CUDA

- Classification Trees
    - Depth First - each CUDA SM/block takes ownership of a single parameter and use the warp to choose optimal split. This provides maximum speedup at early splits.
    - Breadth First - each CUDA SM/block is tasked with generating the full tree. This is more effective when the initial splits have already happened. 
- Note that generating sorted arrays via indexing a permute array (i.e. `a[asort[i]]`) violates data locality and comes with a stiff memory penalty
- Boosting
    - Build one tree at a time with usually just 1 split. Depth first works best here, allows for observation reweighting
- Bagging
    - Build many trees simultaneously. Breadth First approaches work best with each SM building a tree. This can be combined with hybrid approaches, where the first few splits (with most data) are done depth first before going breadth first.

### Problems
    
- Performance is only a slight improvemnt for xgboost (3-10x) 
- Much greater for random forests (can construct multiple trees simultaneously). Benchmarks vary and it's been mostly mediocre (I'd like to see the rapids benchmark upon release) 
- Speedup is greatest when the features are few


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
        - Large models can have 65 million parameters = 1 Gb with Float16 rep. This is not an insignificant part of GPU RAM (also overhead for adaptive gradient descents)
2. Multi GPU
    - Video gamers knew this wasn't great years ago
    - Non trivial algorithms required - magnitude of difficulty increases
    - It's significantly easier to run multiple single GPU experiments
    - Performance scaling can be poor, even within a single machine. 
        - Convolutions scale well, others not so well (1 GPU may be fastest)
    - Many ML algorithms have identifiability which requires some restrictions and assumptions (HOGWILD for one)
3. Sequential Algorithms
    - XGBoost benchmarks show only a 4x speedup
    - MCMC/HMC doesn't benefit from CUDA cores
    - EM iterations still need to be done in sequence (As does Gradient Descent! GPUs are only speeding up the objective function evaluation)
    - Time series analysis tends to use error in previous prediction as a feature. Not very tensor-esque
    - RNNs training benefit is much reduced for this reason (Without NVIDIA provided code, these were slower than the CPU). 6x for backprop, 140x for forward prop
4. Algorithm Changes
    - Algorithms such as DBScan rely on special data structures (R* trees) which required completely new ways of implementing this algorithm
    - XGBoost (trees in general) and KNN require fundamental algorithmic changes to work correctly.
    - Spatial Convolutions are implemented very differently on a GPU than a CPU.
    - Memory is expensive, compute is cheap on a GPU. CPU code tends to be the other way.

5. Technical Debt
    - Compute power is not automatically available without the software. We're limited to what is available. Nvidia is good and proactive, but if you're off the beaten track, they're not very helpful
    - CUDA is non trivial to program in. Most researchers will need a large engineering team to get anything out of their GPUs. 
    - Requires 

6. No real ability to stream (i.e. small amounts continuously)
    - Latency on input - consider running a neural network forward on visual or auditory data as it comes in. GPUs are not good due to the high latency on CPU - GPU memory latency
    - Reinforcement Learning is a good example - data is collected in a simulated environment, batched, sent to GPU to train neural nets. 

# Conclusions

## My Best Practices with GPUs (Personal Opinion)

- Personal use
    - Your video game card
    - Previous gen card
- Check what algorithms you're using before buying. There is more to ML than deep learning.
    - You might still save in power costs.
    - You might still get a speedup despite not being obviously parallelisable
    - it's easier than dealing with clusters
- Device RAM is the commonly biggest limitation. It's cheaper to buy the one with the most RAM than it is to pay your ML Engineer/DS to fiddle with smaller GPUs
- Buying your own hardware tends to pay itself of quickly vs cloud offerings. Though there can be a lot of overhead in managing hardware (especially at mid scale)
    - Much easier to do experiments on local hardware since it avoids data transfer issues.
    - Using GPUs to deliver inference is always going to be cloud based
- Multi GPU
    - Multi-GPU setups are best utilised doing individual contained experiments on each GPU instead of multiple GPU training.
    - Training on many GPUs should wait till your team has a mature tech stack and a decent engineering team (lot's of debugging re utilization)
    - Heat is a non trivial issue. Homemade GPU boxes never take this into account. 

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

# Asides

## Tensor Cores and TPUs

- Use 8 bit integer arithmetic to perform calculations - much more efficient than floating point.
- Software slowly coming out to use these cores
- Mostly optimised for Tensor operations (i.e. matmul) found in neural nets.
- Even more specific and less generalised than a GPU
- Akin to a Bitcoin ASIC - very good at one thing
