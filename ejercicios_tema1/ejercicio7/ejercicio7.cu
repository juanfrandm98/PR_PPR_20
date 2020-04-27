#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 40;

__global__ void MatAdd( float *A, float *B, float *C, int N ) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i * N + j;

  if( i < N && j < N )
    C[index] = A[index] + B[index];

}

__global__ void MatAddFila( float *A, float *B, float *C, int N ) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int fila = N * j;

  if( j < N )
    for( int i = fila; (i - fila) < N; i++ )
      C[i] = A[i] + B[i];

}

__global__ void MatAddColumna( float *A, float *B, float *C, int N ) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if( j < N )
    for( int i = j; (i / N) < N; i += N )
      C[i] = A[i] + B[i];

}

int main() {

  int i;
  const int NN = N * N;

  // Variables para medir el tiempo
  high_resolution_clock::time_point tantes1, tantes2, tantes3;
  high_resolution_clock::time_point tdespues1, tdespues2, tdespues3;
  duration <double> transcurrido1, transcurrido2, transcurrido3;

  cout << "/////////////////////////////////////////////////" << endl;
  cout << "/// Suma de Matrices usando CUDA              ///" << endl;
  cout << "/////////////////////////////////////////////////" << endl;

  // Punteros a memoria host
  float *A = (float*) malloc( NN * sizeof(float) );
  float *B = (float*) malloc( NN * sizeof(float) );
  float *C = (float*) malloc( NN * sizeof(float) );

  // Punteros a memoria device
  float *A_d; float *B_d; float *C_d;
  cudaMalloc( (void **) &A_d, sizeof(float) * NN );
  cudaMalloc( (void **) &B_d, sizeof(float) * NN );
  cudaMalloc( (void **) &C_d, sizeof(float) * NN );

  // Inicialización de los vectores
  for( i = 0; i < NN; i++ ) {
    A[i] = (float)  i;
    B[i] = (float) -i;
  }

  // Copiar datos de la memoria host a device
  cudaMemcpy( A_d, A, sizeof(float) * NN, cudaMemcpyHostToDevice );
  cudaMemcpy( B_d, B, sizeof(float) * NN, cudaMemcpyHostToDevice );

  // Configuración de ejecución
  dim3 threadsPerBlock( 16, 16 );
  dim3 numBlocks( ceil( (float) (N) / threadsPerBlock.x ),
                  ceil( (float) (N) / threadsPerBlock.y ) );

  // Ejecución del kernel
  tantes1 = high_resolution_clock::now();
  MatAdd <<< numBlocks, threadsPerBlock >>> ( A_d, B_d, C_d, N );
  tdespues1 = high_resolution_clock::now();
  transcurrido1 = duration_cast<duration<double>> ( tdespues1 - tantes1 );

  // copiar datos de la memoria device a host
  cudaMemcpy( C, C_d, sizeof(float) * NN, cudaMemcpyDeviceToHost );

  // Resultados:
  for( i = 0; i < NN; i++ )
    printf( "c[%d]=%f\n", i, C[i] );

  // Ejecución del kernel
  tantes2 = high_resolution_clock::now();
  MatAddFila <<< numBlocks, threadsPerBlock >>> ( A_d, B_d, C_d, N );
  tdespues2 = high_resolution_clock::now();
  transcurrido2 = duration_cast<duration<double>> ( tdespues2 - tantes2 );

  // copiar datos de la memoria device a host
  cudaMemcpy( C, C_d, sizeof(float) * NN, cudaMemcpyDeviceToHost );

  // Resultados:
  for( i = 0; i < NN; i++ )
    printf( "c[%d]=%f\n", i, C[i] );

  // Ejecución del kernel
  tantes3 = high_resolution_clock::now();
  MatAddColumna <<< numBlocks, threadsPerBlock >>> ( A_d, B_d, C_d, N );
  tdespues3 = high_resolution_clock::now();
  transcurrido3 = duration_cast<duration<double>> ( tdespues3 - tantes3 );

  // copiar datos de la memoria device a host
  cudaMemcpy( C, C_d, sizeof(float) * NN, cudaMemcpyDeviceToHost );

  // Resultados:
  for( i = 0; i < NN; i++ )
    printf( "c[%d]=%f\n", i, C[i] );

  // Mostrar tiempos:
  cout << "Tiempo de ejecución del algoritmo inicial: " << transcurrido1.count()
  << " segundos." << endl;
  cout << "Tiempo de ejecución del algoritmo por filas: " << transcurrido2.count()
  << " segundos." << endl;
  cout << "Tiempo de ejecución del algoritmo por columnas: " << transcurrido3.count()
  << " segundos." << endl;

  // Liberar memoria
  free(A); free(B); free(C);
  cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

}
