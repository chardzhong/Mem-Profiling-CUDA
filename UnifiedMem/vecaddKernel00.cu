// vecAddKernel00.cu
// This Kernel adds two Vectors A and B in C on GPU
// without using coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N, int opsperthread)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x*opsperthread;
    int end = idx+opsperthread
    for(int i = idx; i<end; i++){
        if(i<N)
            C[i] = A[i] + B[i];
    }
}
