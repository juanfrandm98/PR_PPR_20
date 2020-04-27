#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

using namespace std;

#define BlockSize 128
#define Bsize_maximum 128

// -------------------------------------------------------------------------- //

double cpuSecond() {

  struct timeval tp;
  gettimeofday( &tp, NULL );

  return( (double) tp.tv_sec + (double) tp.tv_usec * 1e-6 );

}

// -------------------------------------------------------------------------- //

__global__ void floyd_unid( int * M, const int nverts, const int k ) {

  int ij = threadIdx.x + blockDim.x * blockIdx.x;

  if( ij < nverts * nverts ) {

    int Mij = M[ij];
    int i = ij / nverts;
    int j = ij - i * nverts;

    if( i != j && i != k && j != k ) {

      int Mikj = M[i * nverts + k] + M[k * nverts + j];
      Mij = ( Mij > Mikj ) ? Mikj : Mij;
      M[ij] = Mij;

    }

  }

}

// -------------------------------------------------------------------------- //

__global__ void floyd_dosd( int * M, const int nverts, const int k ) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if( i < nverts && j < nverts ) {

    int ij = i * nverts + j;
    int Mij = M[ij];

    if( i != j && i != k && j != k ) {

      int Mikj = M[i * nverts + k] + M[k * nverts + j];
      Mij = ( Mij > Mikj ) ? Mikj : Mij;
      M[ij] = Mij;

    }

  }

}

// -------------------------------------------------------------------------- //

__global__ void reduceMax( int * V_in, int * V_out, const int nverts ) {

  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = ( ( i < nverts ) ? V_in[i] : 0.0f );

  if( sdata[tid] == 1000000 )
    sdata[tid] = 0.0f;

  __syncthreads();

  for( int s = blockDim.x / 2; s > 0; s >>= 1 ) {

    if( tid < s )
      if( sdata[tid] < sdata[tid + s] )
        sdata[tid] = sdata[tid + s];

    __syncthreads();

  }

  if( tid == 0 ) {
    V_out[blockIdx.x] = sdata[0];
  }

}

// -------------------------------------------------------------------------- //

int main( int argc, char *argv[] ) {

  if( argc != 2 ) {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return(-1);
  }

  // Obtenemos datos de la GPU
  int devID;
  cudaDeviceProp props;
  cudaError_t err;

  err = cudaGetDevice( &devID );

  if( err != cudaSuccess )
    cout << "Error al mostrar los datos de la GPU." << endl;

  cudaGetDeviceProperties( &props, devID );
  printf( "Device %d: \"%s\" with Compute %d. %d capability\n\n",
          devID, props.name, props.major, props.minor );

  // Lectura del grafo pasado como parámetro
  Graph G;
  G.lee( argv[1] );

  // Preparación de las matrices
  const int nverts  = G.vertices;
  const int niters  = nverts;
  const int nverts2 = nverts * nverts;

  int *c_Out_M = new int[nverts2];
  int size = nverts2 * sizeof(int);
  int *d_In_M = NULL;

  err = cudaMalloc( (void **) &d_In_M, size );
  if( err != cudaSuccess )
    cout << "Error en la reserva - cudaMalloc." << endl;

  int *A = G.Get_Matrix();

  // Fase 1: Ejecución en GPU

  double t1 = cpuSecond();

  err = cudaMemcpy( d_In_M, A, size, cudaMemcpyHostToDevice );
  if( err != cudaSuccess )
    cout << "Error en la copia de la matriz a GPU - cudaMemcpy." << endl;

  //int threadsPerBlock = blocksize;
  //int blocksPerGrid = ( nverts2 + threadsPerBlock - 1 ) / threadsPerBlock;
  dim3 threadsPerBlock( 16, 16 );
  dim3 numBlocks( ceil( (float) (nverts) / threadsPerBlock.x ),
                  ceil( (float) (nverts) / threadsPerBlock.y ) );

  // Ejecución de las iteraciones - kernel
  for( int k = 0; k < niters; k++ ) {

    floyd_dosd <<< numBlocks, threadsPerBlock >>> ( d_In_M, nverts, k );

    err = cudaGetLastError();
    if( err != cudaSuccess ) {
      fprintf( stderr, "Failed to launch kernel! ERROR = %d\n", err );
      exit( EXIT_FAILURE );
    }

  }

  cudaMemcpy( c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();

  double Tgpu = cpuSecond() - t1;

  cout << "Tiempo gastado en GPU = " << Tgpu << endl;

  // Fase 2: Ejecución en CPU

  t1 = cpuSecond();

  int inj, in, kn;

  for( int k = 0; k < niters; k++ ) {

    kn = k * nverts;

    for( int i = 0; i < nverts; i++ ) {

      in = i * nverts;

      for( int j = 0; j < nverts; j++ )

        if( i != j && i != k && j != k ) {
          inj = in + j;
          A[inj] = min( A[in + k] + A[kn + j], A[inj] );
        }

    }

  }

  double t2 = cpuSecond() - t1;

  cout << "Tiempo gastado en CPU = " << t2 << endl << endl;

  cout << "Ganancia = " << t2 / Tgpu << endl;

  for( int i = 0; i < nverts; i++ )
    for( int j = 0; j < nverts; j++ )
      if( abs( c_Out_M[i * nverts + j] - G.arista( i, j ) ) > 0 )
        cout << "Error (" << i << "," << j << ") -> " << c_Out_M[i * nverts + j]
             << "..." << G.arista( i, j ) << endl;

  /*
  cout << "Matriz resultado secuencial:" << endl;

  for( int i = 0; i < nverts; i++ ) {
    cout << i << ": ";
    for( int j = 0; j < nverts; j++ )
      cout << A[i * nverts + j] << " ";
    cout << endl;
  }
  */
  /*
  cout << endl << endl << "Matriz resultado GPU:" << endl;

  for( int i = 0; i < nverts; i++ ) {
    cout << i << ": ";
    for( int j = 0; j < nverts; j++ )
      cout << c_Out_M[i * nverts + j] << " ";
    cout << endl;
  }
  */

  //dim3 dimBlock( Bsize_maximum );
  //dim3 dimGrid ( ceil((float(N)/(float)dimBlock.x)) );

  dim3 threadsPerBlock2( Bsize_maximum, 1 );
  dim3 numBlocks2 ( ceil( (float) nverts2 / threadsPerBlock2.x ), 1 );
  int smemSize = nverts2 * sizeof(int);

  int *C = new int[numBlocks2.x];
  int *C_D;
  cudaMalloc( (void**) &C_D, numBlocks2.x * sizeof(int) );

  reduceMax <<<numBlocks2, threadsPerBlock2, smemSize>>> ( d_In_M, C_D, nverts2 );

  err = cudaGetLastError();
  if( err != cudaSuccess ) {
    fprintf( stderr, "Failed to launch reduction kernel! ERROR = %d\n", err );
    exit( EXIT_FAILURE );
  }

  cudaMemcpy( C, C_D, size, cudaMemcpyDeviceToHost );

  int max = 0.0f;

  for( int i = 0; i < numBlocks2.x; i++ )
    if( max < C[i] )
      max = C[i];

  /*
  for( int i = 0; i < nverts2; i++ ){
    cout << "C[" << i << "]=" << C[i] << endl;
  }
  */

  cout << endl << "Camino máximo GPU = " << max << endl;

  int maxcpu = 0;

  for( int i = 0; i < nverts2; i++ )
    if( A[i] > maxcpu && A[i] != 1000000 )
      maxcpu = A[i];

  cout << "Camino máximo CPU = " << maxcpu << endl;

}
