
#define MAXLEN 1024

__global__ void dot_product(float *x, float *y, int len){
    __shared__ float s[MAXLEN]

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    x[index] += y[index];
    __syncthreads();




}