#include "mpi.h"
#include <iostream>

using namespace std;

int main( int argc, char *argv[] ) {

  int rank, size, value;
  MPI_Status estado;

  MPI_Init( &argc, &argv );

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if( rank == 0 || rank == 1 )
    value = rank;

  MPI_Comm comm;
  int color = rank % 2;

  MPI_Comm_split( MPI_COMM_WORLD, color, rank, &comm );

  MPI_Bcast( &value, 1, MPI_INT, 0, comm );

  cout << "Proceso " << rank << ", el valor compartido es " << value << endl;

  MPI_Finalize();

  return 0;

}
