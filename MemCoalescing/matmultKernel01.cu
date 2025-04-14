// matmultKernel01.cu
// Multiplies two matrices using CUDA: A x B = C
// Threads compute 4 values in a pass
#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes FOUR block sized sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row * 2 + BLOCK_SIZE * block_col * 2];

  // Each thread computes FOUR elements of Csub in its copy of CValues
  float Cvalues[4] = {0,0,0,0};

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / (BLOCK_SIZE*2)); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row * 2 + BLOCK_SIZE * m * 2];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m * 2 + BLOCK_SIZE * block_col * 2];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads FOUR ELEMENTs of ASub and FOUR of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[BLOCK_SIZE*2][BLOCK_SIZE*2];
    __shared__ float shared_B[BLOCK_SIZE*2][BLOCK_SIZE*2];

    // Each thread copies FOUR elements of shared_A and FOUR elements of shared_B
    // upper left block and lower left block
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    shared_A[thread_row + BLOCK_SIZE][thread_col] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];
    shared_B[thread_row + BLOCK_SIZE][thread_col] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col];
    //upper right block and lower right block
    shared_A[thread_row][thread_col+BLOCK_SIZE] = Asub[thread_row * A.stride + thread_col+BLOCK_SIZE];
    shared_B[thread_row][thread_col+BLOCK_SIZE] = Bsub[thread_row * B.stride + thread_col+BLOCK_SIZE];
    shared_A[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col+BLOCK_SIZE];
    shared_B[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col+BLOCK_SIZE];
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B for all FOUR blocks
    // computing FOUR Cvalues by accumulation, one for each block in Csub
#pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e){
        //Upper left
       Cvalues[0] += shared_A[thread_row][e] * shared_B[e][thread_col];
       //Upper right
       Cvalues[1] += shared_A[thread_row][e+BLOCK_SIZE] * shared_B[e+BLOCK_SIZE][thread_col];
       //Lower left
       Cvalues[2] += shared_A[thread_row+BLOCK_SIZE][e] * shared_B[e][thread_col+BLOCK_SIZE];
       //Lower right
       Cvalues[3] += shared_A[thread_row+BLOCK_SIZE][e+BLOCK_SIZE] * shared_B[e+BLOCK_SIZE][thread_col+BLOCK_SIZE];
    }
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  //Upper left
  Csub[thread_row * C.stride + thread_col] = Cvalues[0];
  //Upper right
  Csub[thread_row * C.stride + (thread_col+BLOCK_SIZE)] = Cvalues[1];
  //Low left
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col] = Cvalues[2];
  //Low right
  Csub[(thread_row + BLOCK_SIZE)* C.stride + (thread_col + BLOCK_SIZE)] = Cvalues[3];
}

