/*
Francisco Rodriguez Jimenez
cazz@correo.ugr.es
nvcc - The NVIDIA CUDA Compiler
cuobjdump - The NVIDIA CUDA Object Utility
nvdisasm - The NVIDIA CUDA disassembler
nvprune - The NVIDIA CUDA Prune Tool
nsight - NVIDIA NSight, Eclipse Edition
nvvp - The NVIDIA CUDA Visual Profiler
nvprof - The NVIDIA CUDA Command-Line Profiler
cuda-memcheck - The NVIDIA CUDA Check Tool
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"

using namespace std;

__host__ void check_CUDA_Error(const char *mensaje){
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
        exit(EXIT_FAILURE);
    }
}

__global__ void reduceSum(int *d_V, int *Out, int N, int smen){
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    // printf("\nTAM MEMORIA: %d | tid: %d -> id: %d",smen, tid, i);

	sdata[tid] = ((i < N/2) ? d_V[i] + d_V[i+blockDim.x] : 0.0f);
    __syncthreads();
    
	for (int s = (blockDim.x/2); s > 0; s >>= 1) {
		if (tid < s) {
            sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
    }

    if(tid == 0){
        Out[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv){
    if(argc != 2){
        cout << "Error de sintaxis: ejer8 <TAM>" << endl;
        return(EXIT_FAILURE);
    }

    const int TAM = atoi(argv[1]);
    //Punteros memoria host
    int *vector_entrada, *host_o;
    //Punteros memoria device
    int *device_i, *device_o;

    //Reserva de memoria host
    vector_entrada = new int[TAM];
    //Reserva de memoria device
    cudaMalloc((void **) &device_i, TAM * sizeof(int));
    check_CUDA_Error("Error en la reserva del device");
    
    //Inicialización vector
    for(int i = 0 ; i < TAM; ++i){
        vector_entrada[i] = 1;
    }
      
    cout << "VECTOR ENTRADA: " << endl;
    for(int i = 0 ; i < TAM; ++i){
        cout << vector_entrada[i] << " ";
    }

    //Copia de host a device
    cudaMemcpy(device_i, vector_entrada, sizeof(int)*TAM, cudaMemcpyHostToDevice);
    check_CUDA_Error("Errir en la copia del host al device");

    //Preparo y lanzo el kernel
    dim3 threadsPerBlock(TAM);
    dim3 numBlocks(ceil((float)TAM / threadsPerBlock.x));
    int smemSize = threadsPerBlock.x * sizeof(int);
    cudaMalloc((void **) &device_o, numBlocks.x * sizeof(int));
    host_o = new int[numBlocks.x];
    reduceSum<<<numBlocks, threadsPerBlock, smemSize>>>(device_i, device_o, TAM, threadsPerBlock.x);

    cudaDeviceSynchronize();

    //Copio el resultado de device a host
    cudaMemcpy(host_o, device_o, sizeof(int)*numBlocks.x, cudaMemcpyDeviceToHost);
    
    int suma = 0;
    cout << "\nVECTOR RESULTADO: " << endl;
    for(int i = 0 ; i < numBlocks.x; ++i){
        cout << host_o[i] << " ";
        suma += host_o[i];
    }

    cout << "\n.....................\nRESULTADO FINAL: " << suma << endl;


    delete [] vector_entrada;
    delete [] host_o;
    cudaFree(device_i);
    cudaFree(device_o);
	
	return EXIT_SUCCESS;
}
