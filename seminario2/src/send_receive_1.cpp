#include "mpi.h"
#include <iostream>

using namespace std;

int main( int argc, char *argv[] ) {

  int rank, size, value;
  MPI_Status estado;

  MPI_Init( &argc, &argv );

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if( rank == 0 || rank == 1 ) {

    value = rank;

  } else {

    MPI_Recv( &value, 1, MPI_INT, rank - 2, 0, MPI_COMM_WORLD, &estado );

    cout << "Proceso " << rank << " ha recibido " << value << " del proceso "
         << estado.MPI_SOURCE << endl;

  }

  if( rank + 2 < size )
    MPI_Send( &value, 1, MPI_INT, rank + 2, 0, MPI_COMM_WORLD );

  MPI_Finalize();

  return 0;

}
