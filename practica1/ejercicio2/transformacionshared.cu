#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

#define BlockSize 128

// -------------------------------------------------------------------------- //

__global__ void calculaC( const float * A, const float * B, float * C, const int NBlocks, const int Bsize ) {

  extern __shared__ float sAdata[];
  extern __shared__ float sBdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sAdata[tid] = A[i];
  sBdata[tid] = B[i];
  __syncthreads();

  if( tid < NBlocks * Bsize ) {

    for( int j = 0; j < Bsize; j++ ) {

      float a = sAdata[j] * i;

      if( (int) ceil(a) % 2 == 0 )
        C[i] = a + sBdata[j];
      else
        C[i] = a - sBdata[j];

    }

  }

}

// -------------------------------------------------------------------------- //

__global__ void calculaD( const float * C, float * D, const int NBlocks, const int Bsize ) {

  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = ( (i < NBlocks * Bsize) ? C[i] : 0.0f );
  __syncthreads();

  for( int s = blockDim.x / 2; s > 0; s >>= 1 ) {
    if( tid < s ) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if( tid == 0 )
    D[blockIdx.x] = sdata[0];

}

// -------------------------------------------------------------------------- //

__global__ void calculaMaxC( const float * C, float * max, const int NBlocks, const int Bsize ) {

  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = ( (i < NBlocks * Bsize) ? C[i] : 0.0f );
  __syncthreads();

  for( int s = blockDim.x / 2; s > 0; s >>= 1 ) {

    if( tid < s )
      if( sdata[tid] < sdata[tid + s] )
        sdata[tid] = sdata[tid + s];

    __syncthreads();

  }

  if( tid == 0 )
    max[blockIdx.x] = sdata[0];

}

// -------------------------------------------------------------------------- //

int main( int argc, char *argv[] ) {

  int Bsize, NBlocks;

  if( argc != 3 ) {
    cout << "Uso: " << argv[0] << " <Num_bloques> <Tam_bloque>" << endl;
    return(0);
  }

  NBlocks = atoi( argv[1] );
  Bsize   = atoi( argv[2] );

  const int N     = Bsize * NBlocks;
  const int size  = N * sizeof(float);
  const int dsize = NBlocks * sizeof(float);

  // Punteros a memoria host
  float *A, *B, *Ccpu, *Dcpu, *Cgpu, *Dgpu, *mxgpu;

  // Colocando punteros en el host
  A = new float[N];
  B = new float[N];
  Ccpu = new float[N];
  Dcpu = new float[NBlocks];
  Cgpu = new float[N];
  Dgpu = new float[NBlocks];
  float mxcpu;
  mxgpu = new float[N];

  // Inicializando vectores A y B
  for( int i = 0; i < N; i++ ) {
    A[i] = 5;
    B[i] = 5;
  }

  // ------------------------------- FASE CPU ------------------------------- //
  double t1cpu = clock();

  for( int k = 0; k < NBlocks; k++ ) {

    int istart = k * Bsize;
    int iend   = istart + Bsize;

    Dcpu[k] = 0.0;

    for( int i = istart; i < iend; i++ ) {

      Ccpu[i] = 0.0;

      for( int j = istart; j < iend; j++ ) {

        float a = A[j] * i;

        if( (int) ceil(a) % 2 == 0 )
          Ccpu[i] = a + B[j];
        else
          Ccpu[i] = a - B[j];

      }

      Dcpu[k] += Ccpu[i];
      mxcpu = ( i == 1 ) ? Ccpu[0] : max( Ccpu[i], mxcpu );

    }

  }

  double t2cpu = clock();

  double tcpu = ( t2cpu - t1cpu ) / CLOCKS_PER_SEC;

  // ------------------------------- FASE GPU ------------------------------- //


  // Punteros a memoria device
  float *a_d, *b_d, *c_d, *d_d, *max_d;

  // Colocando arrays en device
  cudaMalloc( (void **) &a_d, size );
  cudaMalloc( (void **) &b_d, size );
  cudaMalloc( (void **) &c_d, size );
  cudaMalloc( (void **) &d_d, dsize );
  cudaMalloc( (void **) &max_d, size );

  // Copiando los datos de memoria host a device
  cudaMemcpy( a_d, A, size, cudaMemcpyHostToDevice );
  cudaMemcpy( b_d, B, size, cudaMemcpyHostToDevice );

  // Lanzamiento del Kernel que calcula C (memoria compartida)
  dim3 threadsPerBlockC( Bsize, 1 );
  dim3 numBlocksC( NBlocks, 1 );
  int smemSize = 2 * Bsize * sizeof(float);

  double t1gpu = clock();

  calculaC<<<numBlocksC, threadsPerBlockC, smemSize>>>( a_d, b_d, c_d, NBlocks, Bsize );

  // Lanzamiento del Kernel que calcula D
  int smemSize2 = Bsize * sizeof(float);
  calculaD<<<numBlocksC, threadsPerBlockC, smemSize2>>>( c_d, d_d, NBlocks, Bsize );

  // Lanzamiento del Kernel que calcula el máximo
  calculaMaxC<<<numBlocksC, threadsPerBlockC, smemSize2>>>( c_d, max_d, NBlocks, Bsize );

  double t2gpu = clock();

  // Copiando los datos de memoria device a host
  cudaMemcpy( Cgpu, c_d, size, cudaMemcpyDeviceToHost );
  cudaMemcpy( Dgpu, d_d, dsize, cudaMemcpyDeviceToHost );
  cudaMemcpy( mxgpu, max_d, size, cudaMemcpyDeviceToHost );

  float mx2 = 0.0f;

  for( int i = 0; i < NBlocks; i++ )
    if( mx2 < mxgpu[i] )
      mx2 = mxgpu[i];

  double tgpu = ( t2gpu - t1gpu ) / CLOCKS_PER_SEC;

  // ------------------------------ RESULTADOS ------------------------------ //

  cout << endl << "///// RESULTADOS EN CPU /////" << endl;
  cout << "------------------------------------" << endl;
  //for( int i = 0; i < N; i++ )
    //cout << "C[" << i << "] = " << Ccpu[i] << endl;
  cout << "------------------------------------" << endl;
  //for( int k = 0; k < NBlocks; k++ )
    //cout << "D[" << k << "] = " << Dcpu[k] << endl;
  cout << "------------------------------------" << endl;

  cout << endl << "El valor máximo en C es: " << mxcpu << endl;
  cout << "Tiempo gastado en CPU: " << tcpu << endl << endl;

  cout << endl << "///// RESULTADOS EN GPU /////" << endl;
  cout << "------------------------------------" << endl;
  //for( int i = 0; i < N; i++ )
    //cout << "C[" << i << "] = " << Cgpu[i] << endl;
  cout << "------------------------------------" << endl;
  //for( int k = 0; k < NBlocks; k++ )
    //cout << "D[" << k << "] = " << Dgpu[k] << endl;
  cout << "------------------------------------" << endl;

  cout << endl << "El valor máximo en C es: " << mx2 << endl;
  cout << "Tiempo gastado en GPU: " << tgpu << endl << endl;

  cout << "Ganancia = " << tcpu / tgpu << endl;
  cout << "Error en el máximo = " << mxcpu - mx2 << endl << endl;

}
