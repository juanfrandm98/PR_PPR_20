#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;

int main( int argc, char * argv[] ) {

  MPI_Init( &argc, &argv );

  Graph G;
  int nverts, rank, size;

  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  if( rank == 0 ) {

    // El proceso 0 comprueba que se le ha pasado un argumento
    if( argc != 2 ) {
      cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
      return(-1);
    }

    // Si el argumento es correcto, lee el grafo de entrada
    G.lee( argv[1] );
    nverts = G.vertices;

  }

  // Se realiza un Broadcast del tamaño de la matriz grande
  MPI_Bcast( &nverts, 1, MPI_INT, 0, MPI_COMM_WORLD );

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 1                                            //
  //    DISTRIBUCIÓN INICIAL DE LA MATRIZ DE ENTRADA      //
  //                                                      //
  //////////////////////////////////////////////////////////

  MPI_Datatype MPI_BLOQUE;

  int raiz_P = sqrt( size );
  int tam = nverts / raiz_P;

  // Creación del buffer de envío para almacenar los datos empaquetados
  buf_envio = reservar_vector( nverts * nverts );

}
