#include <stdio.h>
#include <cuda.h>

#define NBIN 1000000
#define NUM_BLOCK 13
#define NUM_THREAD 192

int tid;
float pi = 0;

__global__ void cal_pi( float *sum, int nbin, float step, int nthreads, int nblocks ) {

  int i;
  float x;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for( i = idx; i < nbin; i += nthreads * nblocks ) {
    x = ( i - 0.5 ) * step;
    sum[idx] += 4.0 / ( 1.0 + x * x );
  }

}

int main( void ) {

  dim3 dimGrid( NUM_BLOCK, 1, 1 );
  dim3 dimBlock( NUM_THREAD, 1, 1 );
  float *sumHost, *sumDev;

  float step = 1.0 / NBIN;
  size_t size = NUM_BLOCK * NUM_THREAD * sizeof( float );

  sumHost = (float *) malloc( size );
  cudaMalloc( (void **) &sumDev, size );

  cudaMemset( sumDev, 0, size );

  cal_pi <<< dimGrid, dimBlock >>> ( sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK );

  cudaMemcpy( sumHost, sumDev, size, cudaMemcpyDeviceToHost );

  for( tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++ )
    pi += sumHost[tid];

  pi *= step;

  printf( "PI = %f\n", pi );

  free( sumHost );
  cudaFree( sumDev );

  return 0;

}
