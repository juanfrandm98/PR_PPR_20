#include "mpi.h"
#include <iostream>
#include <vector>

using namespace std;

int main( int argc, char *argv[] ) {

  int tam, rank, size, istart, iend;

  MPI_Init( &argc, &argv );

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if( argc < 2 ) {

    if( rank == 0 ) {

      cout << "No se ha especificado el número de elementos, múltiplo de la "
           << "cantidad de entrada, por defecto será " << size * 100;
      cout << endl << "Uso: " << argv[0] << " <cantidad>" << endl;

    }

    tam = size * 100;

  } else {

    tam = atoi( argv[1] );

    if( tam < size )
      tam = size;
    else {

      int i = 1, num = size;

      while( tam > num ) {
        ++i;
        num = size * i;
      }

      if( tam != num ) {
        if( rank == 0 )
          cout << "Cantidad cambiada a " << num << endl;
        tam = num;
      }

    }

  }

  // Creación y relleno del vector A
  vector<long> VectorA, VectorALocal;

  VectorA.resize( tam, 0 );
  VectorALocal.resize( tam / size, 0 );

  if( rank == 0 )
    for( long i = 0; i < tam; i++ )
      VectorA[i] = i + 1;

  // Repartimos el vector A
  MPI_Scatter( &VectorA[0], tam / size, MPI_LONG, &VectorALocal[0], tam / size,
               MPI_LONG, 0, MPI_COMM_WORLD );

  // Creación y relleno de los vectores locales B
  vector<long> VectorBLocal;

  VectorBLocal.resize( tam / size, 0 );

  istart = rank * ( tam / size );
  iend = istart + ( tam / size );
  int j = 0;

  for( long i = istart; i < iend; i++ ) {
    VectorBLocal[j] = ( i + 1 ) * 10;
    j++;
  }

  // Cálculo de la multiplicación escalar entre vectores
  long producto = 0;

  for( long i = 0; i < tam / size; i++ )
    producto += VectorALocal[i] * VectorBLocal[i];

  long total;

  // Reunimos los datos en un solo proceso, aplicando una operación aritmética,
  // en este caso, la suma
  MPI_Reduce( &producto, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD );

  if( rank == 0 )
    cout << "Total = " << total << endl;

  MPI_Finalize();

  return 0;

}
