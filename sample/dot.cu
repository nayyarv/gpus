


__global__ void sum(float *x, float *y, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // assume that we have launched enough threads
    // see grid-stride loops for better practice
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}


// Use static allocations for simplicity
#define MAXLEN 1024

__global__ void reduce_sum(float *x, float *out, int n){
    // shared defines a local cache that is fast
    __shared__ float s[MAXLEN]; 

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        s[threadIdx.x] = x[index];
    } else {
        s[threadIdx.x] = 0.0f;
    }

    // ensure the data is ready to reduce
    __syncthreads();

    // CUDA provides functions, this is instructional
    for (int s = blockDim.x/2; s > 0; s>>=1) {
        // Only works for powers of 2
        if (threadIdx.x < s) {
            sarray[threadIdx.x] += sarray[threadIdx.x+s];
        }
        __syncthreads();
    }
    // out will be the length of blockdim
    out[blockIdx.x] = s[0];
}