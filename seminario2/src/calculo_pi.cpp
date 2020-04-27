#include "mpi.h"
#include <iostream>
#include <math.h>

using namespace std;

int main( int argc, char *argv[] ) {

  int n, rank, size, Bsize, istart, iend;
  int mensaje[2];

  double PI25DT = 3.141592653589793238462643;

  double mypi, pi, h, sum;

  MPI_Status estado;

  MPI_Init( &argc, &argv );

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // El proceso 0 es el encargado de pedir el número de iteraciones a ejecutar
  if( rank == 0 ) {

    do {
      cout << "Introduce la precisión del cálculo (n > 0): ";
      cin >> n;
    } while( n <= 0 );

  }

  // El proceso 0 envía a todos los demás procesos el número de iteraciones
  // totales que calcularemos para la aproximación de PI
  MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD );

  // El proceso 0 reparte las ejecuciones
  if( rank == 0 ) {

    Bsize = ceil( (float) n / size );

    istart = 1;
    iend = istart + Bsize;

    int restantes = n - Bsize;
    int aux_inicio, aux_bloque;

    for( int i = 1; i < size; i++ ) {

      aux_inicio = ( i * Bsize ) + 1;

      if( restantes >= Bsize ) {
        aux_bloque = Bsize;
        restantes -= Bsize;
      } else {
        aux_bloque = restantes;
        restantes = 0;
      }

      mensaje[0] = aux_inicio;
      mensaje[1] = aux_bloque;

      // El proceso 0 manda a cada proceso la primera iteración que debe
      // calcular y cuántas iteraciones debe realizar
      MPI_Send( &mensaje, 2, MPI_INT, i, 0, MPI_COMM_WORLD );

    }

  } else {

    MPI_Recv( &mensaje, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &estado );

    istart = mensaje[0];
    iend = istart + mensaje[1];

  }

  // El proceso 0 envía a todos los demás procesos el número de iteraciones
  // totales que calcularemos para la aproximación de PI
  MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD );

  // Cálculo de PI
  h = 1.0 / (double) n;
  sum = 0.0;

  for( int i = istart; i < iend; i++ ) {
    double x = h * ( (double) i - 0.5 );
    sum += ( 4.0 / ( 1.0 + x * x ) );
  }

  mypi = h * sum;

  MPI_Barrier( MPI_COMM_WORLD );

  // Todos los procesos comparten su valor local de PI a través de una reducción
  // de su valor local a todos los procesos
  MPI_Allreduce( &mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  //MPI_Reduce( &mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  //MPI_Bcast( &pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  // Todos los procesos imprimen el valor de PI aproximado
  cout << "Proceso " << rank << ": el valor aproximado de PI es: " << pi << endl;

  // El proceso 0 informa del error
  if( rank == 0 )
    cout << "Proceso 0: PI ha sido calculado con un valor de " <<
            fabs( pi - PI25DT ) << endl;

  MPI_Finalize();

  return 0;

}
