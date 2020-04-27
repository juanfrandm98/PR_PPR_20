#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

#define Bsize_addition 256
#define Bsize_minimum  128

__global__ void reduceSum( float * d_V, int N ) {

  // Vector en memoria compartida para almacenar los datos
  extern __shared__ float sdata[];

  // Cálculo de los índices para acceder al vector
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // Carga de los datos en memoria compartida (dos por hebra)
  sdata[tid] = ( ( i < N ) ? d_V[i] + d_V[i + blockDim.x] : 0.0f );

  __syncthreads();

  // Reducción en memoria compartida
  for( int s = blockDim.x/2; s > 0; s >>= 1 ) {
    if( tid < s )
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Escribir el resultado en memoria principal
  if( tid == 0 )
    d_V[blockIdx.x] = sdata[0];

}

int main( int argc, char *argv[] ) {

  cout << "////////////////////////////////////" << endl;
  cout << "/// Suma de vector por reducción ///" << endl;
  cout << "////////////////////////////////////" << endl;

  int N;

  cout << endl << "Introduce el número de elementos del vector: ";
  cin >> N;

  // Puntero a memoria host
  float *h_V;

  // Puntero a memoria device
  float *d_V;

  // Situar el array h_V en el host
  h_V = (float*) malloc( N * sizeof( float ) );

  // Situar el array d_V en device
  cudaMalloc( (void **) &d_V, sizeof(float) * N );

  // Inicializar el array d
  for( int i = 0; i < N; i++ )
    h_V[i] = (float) 1;

  // Copiar los datos de la memoria host a device
  cudaMemcpy( d_V, h_V, sizeof(float) * N, cudaMemcpyHostToDevice );

  // Configuración de la ejecución
  //dim3 dimBlock( Bsize_addition );
  //dim3 dimGrid( ceil( (float(N)) / (float) dimBlock.x ) );

  // ADD ARRAYS A AND B, STORE RESULT IN C

  dim3 threadsPerBlock( 32 );
  dim3 numBlocks( ceil( ( float ) ( N / 2 ) / threadsPerBlock.x ), 1 );
  int smemSize = threadsPerBlock.x * sizeof( float );

  // Variables para medir el tiempo
  high_resolution_clock::time_point tantes, tdespues;
  duration<double> transcurrido;

  // Ejecución del kernel
  tantes = high_resolution_clock::now();
  reduceSum <<< numBlocks, threadsPerBlock, smemSize >>> ( d_V, N );
  cudaDeviceSynchronize();
  tdespues = high_resolution_clock::now();

  transcurrido = duration_cast<duration<double>> ( tdespues - tantes );

  // Copiar los datos de la memoria device a host
  cudaMemcpy( h_V, d_V, N * sizeof( float ), cudaMemcpyDeviceToHost );

  int suma = 0;

  for( int i = 0; i < numBlocks.x; i++ )
    suma += h_V[i];

  // Resultados:
  cout << endl << "Resultado: " << suma << " / " << h_V[0] << endl;
  cout << "El tiempo empleado es " << transcurrido.count() << " segundos." <<
  endl << endl;

}
