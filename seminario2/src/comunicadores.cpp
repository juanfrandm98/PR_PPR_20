#include "mpi.h"
#include <iostream>
#include <vector>

using namespace std;

int main( int argc, char *argv[] ) {

  int rank, size;

  MPI_Init( &argc, &argv );

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  MPI_Comm comm_inverso, comm;

  int rank_inverso, size_inverso;
  int rank_nuevo, size_nuevo;
  int a, b, c;

  if( rank == 0 ) {
    a = 2000;
    b = 1;
    c = 0;
  } else {
    a = 0;
    b = 0;
    c = 0;
  }

  int color = rank % 2;

  // Creamos los nuevos comunicadores
  MPI_Comm_split( MPI_COMM_WORLD, color, rank, &comm );
  MPI_Comm_split( MPI_COMM_WORLD, 0, -rank, &comm_inverso );

  // Obtenemos los rangos y tamaños de los nuevos comunicadores
  MPI_Comm_rank( comm, &rank_nuevo );
  MPI_Comm_size( comm, &size_nuevo );

  vector<int> valores_impares;
  valores_impares.resize( size_nuevo, 0 );

  if( rank == 1 )
    for( int i = 0; i < size_nuevo; i++ )
      valores_impares[i] = ( ( i + 1 ) * 10 );

  if( ( rank % 2 ) != 0 )
    MPI_Scatter( &valores_impares[0], 1, MPI_INT, &c, 1, MPI_INT, 0, comm );

  MPI_Comm_rank( comm_inverso, &rank_inverso );
  MPI_Comm_size( comm_inverso, &rank_inverso );

  // Enviamos los datos por distintos comunicadores
  MPI_Bcast( &a, 1, MPI_INT, 0, comm );
  MPI_Bcast( &b, 1, MPI_INT, size - 1, comm_inverso );

  cout << "Soy el proceso " << rank << " de " << size << " dentro de MPI_COMM_WORLD,"
       << "\n\tmi rango en COMM_nuevo es " << rank_nuevo << " de " << size_nuevo
       << ", aquí he recibido el valor " << a << ",\n\ten COMM_inverso mi rango es "
       << rank_inverso << " de " << size_inverso << ", aquí he recibido el valor "
       << b << "\n\t, mi valor del vector de impares es " << c << endl << endl;

  MPI_Finalize();

  return 0;

}
