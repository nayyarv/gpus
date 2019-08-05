#include <stdlib.h>
#include <math.h>
// #include <stdio.h>

#define MAXTHREADS MAX_THREADS
#define PI 3.1415926536

__device__ float normalDistribution(float* x, float* mu,
    float* diagonalCovariance, unsigned int dim){
    /*
    x:  individual point being evaluated, x[dim]
    mu: mean of the normal distribution being evaluated  mu[dim]
    diagonalCovariance: for the norm dist  diagonalCovariance[dim]

    dim: dimensionality of the distribution, also
    equal to length of the previous vectors
    Evaluates the normalDistribution on the GPU, for
    diagonal Covariance Matrices only.
    */
    float total = 0;
    float det = 1;
    float finval = 0;
    float denom = 0;
    float temp = 0;

    for (int i = 0; i < dim; ++i)
    {
        temp = (x[i]-mu[i]);
        temp *= temp; // Square it
        total += temp / diagonalCovariance[i];
        //Note this is the stuff that goes inside the normal
        det *= diagonalCovariance[i];
        //TODO: replace with memory implementation instead?
    }


    // printf("temp = %f, det = %f, total = %f\n", temp, det, total);

    total*=-1/2.0;

    finval = expf(total);

    denom = powf(2*PI, dim) * det;

    return (rsqrtf(denom) * finval);
}


__global__ void likelihoodKernel(float *Xpoints, float *means, float *diagCovs,
    float *weights,
    unsigned int dim, unsigned int numPoints, unsigned int numMixtures,
    float* finalLikelihood)
{
    /*
    All 2d arrays are passed in as row major
    Xpoints - 2d array of points, numPoints rows of vectors of dim length
        Xpoints[numPoints][dim]
    Means - 2d array of means, numMixtures rows of vectors of dim
        Means[numMixtures][dim]
    diagCovs - 2d array of cov diagonals, ditto
        diagCovs[numMixtures][dim]
    weights - 1d array of length numMixtures
        weights[numMixtures]

    numPoints is the actual number of points being evaluated
        This is likely to be a subset of what actually needs to be processe
        GridDim*BlockDim > numPoints
    finalLikelihood: Likelihood value that we want to return
        finalLikelihood[blockIdx.x]
    Since threads are usually a power of 2, we have to check if we're out of bounds
    with regards to the data.
    */

    __shared__ float sarray[MAXTHREADS];
    //Should be consistently at the max allowed and easier than dynamic allocation

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.x;

    sarray[threadIndex] = 0;

    __syncthreads();
    //Following CUDA guidelines here for quick reduction
    //TODO: Speed up computation by having a block per mixture?
    // If possible, also allows for marginal updates


    if (index<numPoints) //Check that we're in bounds!
    {
        // Just make sure we have stuff to compute
        // Will contain the id of the x value

        float value = 0;

        for (int i = 0; i < numMixtures; ++i)
        {
            value += weights[i] * normalDistribution(Xpoints+(index*dim), means+(i*dim),
                                                     diagCovs+(i*dim), dim);
        }

        sarray[threadIndex] = logf(value); //Log Likelihood



    }
    else
    {
        sarray[threadIndex] = 0.0f; //I.e it's zero
    }

    // finalLikelihood[threadIndex] = sarray[threadIndex];

    // Reduction
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s>>=1)
    {
        //Only works for powers of 2
        if (threadIndex<s)
        {
            sarray[threadIndex] += sarray[threadIndex+s];
        }
        __syncthreads();
    }


    if (threadIndex==0)
    //Since everything has been synced, sarray[0] now holds our result
    {
        finalLikelihood[blockIdx.x] = sarray[0];
    }


}