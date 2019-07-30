


__globa__ void sum(float *x, float *y, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // assume that we have launched enough threads in this case
    // see grid-stride loops for better practice
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}


// Use static allocations for simplicity
#define MAXLEN 1024

__global__ void dot_product(float *x, float *y, float *out, int n){
    // shared defines a local cache with very fast lookup speeds
    // we do this since we're going to sum this array and will
    // need fast lookup
    __shared__ float s[MAXLEN]; 


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        s[threadIdx.x] = x[index] * y[index];
    } else {
        s[threadIdx.x] = 0.0f;
    }

    // ensure the data is ready to reduce
    __syncthreads();

    // CUDA provides functions for this, but let's manually write this out for fun
    for (int s = blockDim.x/2; s > 0; s>>=1) {
        // Only works for powers of 2
        if (threadIdx.x < s) {
            sarray[threadIdx.x] += sarray[threadIdx.x+s];
        }
        __syncthreads();
    }
    // out will be the length of blockdim. This can then be reduced with another kernel
    // later on
    out[blockIdx.x] = s[0];

}