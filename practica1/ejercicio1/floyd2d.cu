#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

using namespace std;

#define blocksize 1024

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

  cout << endl << endl << "Matriz resultado GPU:" << endl;

  for( int i = 0; i < nverts; i++ ) {
    cout << i << ": ";
    for( int j = 0; j < nverts; j++ )
      cout << c_Out_M[i * nverts + j] << " ";
    cout << endl;
  }
  */

}
