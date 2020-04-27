#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

__global__ void calcularPI( float *sum, long num_steps, float step ) {

  double x;

  for( int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num_steps; i += num_steps ) {
    x = ( i - 0.5 ) * step;
    sum[i] = 4.0 / ( 1.0 + x * x );
  }

}

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

int main() {

  long num_steps;

  cout << "/////////////////////////////////////////////////" << endl;
  cout << "/// Calculo del número PI de utilizando CUDA  ///" << endl;
  cout << "/////////////////////////////////////////////////" << endl;
  cout << endl << "Introduce el número de pasos: ";
  cin >> num_steps;

  float step = 1.0 / num_steps;

  // Puntero a memoria host
  float *h_V;

  // Puntero a memoria device
  float *d_V;

  // Situar el array h_V en el host
  h_V = (float*) malloc( num_steps * sizeof( float ) );

  // Situar el array d_V en device
  cudaMalloc( (void**) &d_V, sizeof( float ) * num_steps );

  // Copiar los datos de la memoria host a device
  cudaMemcpy( d_V, h_V, sizeof(float) * num_steps, cudaMemcpyHostToDevice );

  dim3 threadsPerBlock( 32 );
  dim3 numBlocks( ceil( (float) num_steps / threadsPerBlock.x ), 1 );
  int smemSize = threadsPerBlock.x * sizeof(float);

  // Medición temporal
  high_resolution_clock::time_point tantes, tdespues;
  duration<double> transcurrido;

  // Ejecución del kernel
  tantes = high_resolution_clock::now();
  calcularPI <<< numBlocks, threadsPerBlock, smemSize >>> ( d_V, num_steps, step );
  cudaDeviceSynchronize();
  tdespues = high_resolution_clock::now();

  transcurrido = duration_cast<duration<double>> ( tdespues - tantes );

  reduceSum <<< numBlocks, threadsPerBlock, smemSize >>> ( d_V, num_steps );

  // Copiar los datos de la memoria device a host
  cudaMemcpy( h_V, d_V, numBlocks.x * sizeof( float ), cudaMemcpyDeviceToHost );

  double pi = 0;

  for( int i = 0; i < numBlocks.x; i++ ) {
    //cout << i << "->" << h_V[i] << endl;
    pi += h_V[i];
  }

  pi *= step;

  cout << endl << "Resultado: PI = " << pi << "." << endl;
  cout << "El tiempo empleado es " << transcurrido.count() << " segundos." <<
  endl << endl;

}
