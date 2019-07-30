


__globa__ void sum(float *x, float *y, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // assume that we have launched enough threads in this case
    // see grid-stride loops for better practice
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}


#define MAXLEN 1024
// Use static allocations for simplicity

__global__ void dot_product(float *x, float *y, int len){
    __shared__ float s[MAXLEN]; 
    // shared defines a local cache with very fast lookup speeds
    // we do this 

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    s[index] = x[index] + y[index];
    __syncthreads();


}