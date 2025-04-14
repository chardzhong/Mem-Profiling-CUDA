#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"
// Utility Functions
void Cleanup(bool);
// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 
int cuda; //cuda option

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    int K; // number of values in millions
    int N; //Vector size
    
	// Parse arguments.
    if(argc != 3){
     printf("Usage: %s K cuda\n", argv[0]);
     printf("K is the number of values to be added in the millions.\n");
     printf("Total vector size is k * 1,000,000.\n");
     printf("cuda is the option to use cuda: 0 for cpu, 1 for cuda, 2 for cuda with unified memory\n");
     exit(0);
    } else {
      sscanf(argv[1], "%d", &K);
      sscanf(argv[2], "%d", &cuda);
    }      
   
    
    N = K*1000000;
    printf("Total vector size: %d\n", N); 
    // size_t is the total number of bytes for a vector.
    size_t size = N * sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;   
    }


    //Run on CPU
    if(!cuda){
        printf("CPU:\n");
        // Initialize timer  
        initialize_timer();
        start_timer();
        for( i=0; i<N; ++i ){
            h_C[i] = h_A[i] + h_B[i];
        }
        // Compute elapsed time 
        stop_timer();
        double time = elapsed_time();
        // Report timing data.
        printf( "Time: %lf (sec)\n", time);
        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = h_C[i];
            if (fabs(val - (h_A[i]+h_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
        // Clean up and exit.
        Cleanup(true);
    }


    //CUDA no unified memory
    if(cuda==1){
        // Allocate vectors in device memory.
        cudaError_t error;
        error = cudaMalloc((void**)&d_A, size);
        if (error != cudaSuccess) Cleanup(false);
        error = cudaMalloc((void**)&d_B, size);
        if (error != cudaSuccess) Cleanup(false);
        error = cudaMalloc((void**)&d_C, size);
        if (error != cudaSuccess) Cleanup(false);
        // Copy host vectors h_A and h_B to device vectores d_A and d_B
        error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) Cleanup(false);
        error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) Cleanup(false);

        printf( "CUDA no unified memory:\n");
        printf( "1 block 1 thread\n");
        //one block one thread
        dim3 dimGrid1(1);                    
        dim3 dimBlock1(1);
        int opsperthread = N;
        // Warm up
        AddVectors<<<dimGrid1, dimBlock1>>>(d_A, d_B, d_C, N, opsperthread);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGrid1, dimBlock1>>>(d_A, d_B, d_C, N, opsperthread);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time1 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 1, 1, time1);

        // Copy result from device memory to host memory
        error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) Cleanup(false);

        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = h_C[i];
            if (fabs(val - (h_A[i]+h_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");



        printf( "1 block 256 thread\n");
        //one block one thread                    
        dim3 dimBlock256(256);
        int opsperthread1 = (N+256-1)/256;
        // Warm up
        AddVectors<<<dimGrid1, dimBlock256>>>(d_A, d_B, d_C, N, opsperthread1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGrid1, dimBlock256>>>(d_A, d_B, d_C, N, opsperthread1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time2 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 256, 1, time2);

        // Copy result from device memory to host memory
        error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) Cleanup(false);

        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = h_C[i];
            if (fabs(val - (h_A[i]+h_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");



        printf( "Mult block 256 thread\n");
        int GridWidth = (N+256-1)/256;
        dim3 dimGridMult(GridWidth);

        // Warm up
        AddVectors<<<dimGridMult, dimBlock256>>>(d_A, d_B, d_C, N, 1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGridMult, dimBlock256>>>(d_A, d_B, d_C, N, 1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time3 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 256, GridWidth, time3);

        // Copy result from device memory to host memory
        error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) Cleanup(false);

        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = h_C[i];
            if (fabs(val - (h_A[i]+h_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
        // Clean up and exit.
        Cleanup(true);
    }


    //CUDA with unified memory
    if(cuda==2){
        // Allocate vectors in unified memory.
        cudaError_t error;
        error = cudaMallocManaged((void**)&d_A, size);
        if (error != cudaSuccess) Cleanup(false);
        error = cudaMallocManaged((void**)&d_B, size);
        if (error != cudaSuccess) Cleanup(false);
        error = cudaMallocManaged((void**)&d_C, size);
        if (error != cudaSuccess) Cleanup(false);
        // Initialize unified vectors d_A and d_B
        for(i=0; i<N; ++i){
            d_A[i] = 1.0f;
            d_B[i] = 2.0f;   
        }

        printf( "CUDA with unified memory:\n");
        printf( "1 block 1 thread\n");
        //one block one thread
        dim3 dimGrid1(1);                    
        dim3 dimBlock1(1);
        int opsperthread = N;
        // Warm up
        AddVectors<<<dimGrid1, dimBlock1>>>(d_A, d_B, d_C, N, opsperthread);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGrid1, dimBlock1>>>(d_A, d_B, d_C, N, opsperthread);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time1 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 1, 1, time1);

        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = d_C[i];
            if (fabs(val - (d_A[i]+d_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
        for(i=0; i<N; ++i){
            d_C[i] = 0.0f;;   
        }


        printf( "1 block 256 thread\n");
        //one block one thread                    
        dim3 dimBlock256(256);
        int opsperthread1 = (N+256-1)/256;
        // Warm up
        AddVectors<<<dimGrid1, dimBlock256>>>(d_A, d_B, d_C, N, opsperthread1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGrid1, dimBlock256>>>(d_A, d_B, d_C, N, opsperthread1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time2 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 256, 1, time2);
        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = d_C[i];
            if (fabs(val - (d_A[i]+d_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
        for(i=0; i<N; ++i){
            d_C[i] = 0.0f;;   
        }


        printf( "Mult block 256 thread\n");
        int GridWidth = (N+256-1)/256;
        dim3 dimGridMult(GridWidth);
        // Warm up
        AddVectors<<<dimGridMult, dimBlock256>>>(d_A, d_B, d_C, N,1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddVectors<<<dimGridMult, dimBlock256>>>(d_A, d_B, d_C, N,1);
        error = cudaGetLastError();
        if (error != cudaSuccess) Cleanup(false);
        cudaThreadSynchronize();

        // Compute elapsed time 
        stop_timer();
        double time3 = elapsed_time();
        // Report timing data.
        printf( "Threads: %d Blocks: %d Time: %lf (sec)\n", 256, GridWidth, time3);

        // Verify & report result
        for (i = 0; i < N; ++i) {
            float val = d_C[i];
            if (fabs(val - (d_A[i]+d_B[i])) > 1e-5)
            break;
        }
        printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
        // Clean up and exit.
        Cleanup(true);
    }

}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if(cuda!=0){
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);
        error = cudaThreadExit();
        if(error != cudaSuccess)
            printf("cuda thread exit failed \n");
    }  
    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
    
    if (!noError)
        printf("cuda malloc failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}