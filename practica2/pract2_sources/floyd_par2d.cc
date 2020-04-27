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

  int *A = G.Get_Matrix();

  int bsize1d = nverts / size;
  int bsize2d =

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 1                                            //
  //    DISTRIBUCIÓN INICIAL DE LA MATRIZ DE ENTRADA      //
  //                                                      //
  //////////////////////////////////////////////////////////

  MPI_Datatype MPI_BLOQUE;

  // Calculamos el número de elementos que tendrá cada submatriz (bloque)
  int raiz_P = sqrt( size );
  int tam = nverts / raiz_P;

  // Creación del buffer de envío para almacenar los datos empaquetados
  int *buf_envio = new int[nverts * nverts];
  int tam_buf_envio = sizeof( int ) * nverts * nverts;

  if( rank == 0 ) {

    // Definición del tipo bloque cuadrado, donde cada argumento indica
    //    - tam        -> número de bloques
    //    - tam        -> tamaño de bloque
    //    - nverts     -> separación entre bloques
    //    - MPI_INT    -> tipo de dato utilizado
    //    - MPI_BLOQUE -> tipo de dato creado
    MPI_Type_vector( tam, tam, nverts, MPI_INT, &MPI_BLOQUE );
    MPI_Type_commit( &MPI_BLOQUE );

    // La matriz se empaqueta bloque a bloque en el buffer de envío
    for( int i = 0, posicion = 0; i < size; i++ ) {

      // Cálculo de la posición de comienzo de cada submatriz
      int fila_P = i / raiz_P;
      int columna_P = i % raiz_P;

      comienzo = ( columna_P * tam ) + ( fila_P * tam * tam * raiz_P );

      // Se empaqueta el bloque i, donde cada argumento indica:
      //    - A(comienzo)     -> primer elemento a enviar
      //    - 1               -> número de elementos a enviar
      //    - MPI_BLOQUE      -> tipo de dato a enviar
      //    - buf_envio       -> buffer de envio
      //    - tam_buf_envio   -> tamaño del buffer de envio
      //    - &posicion       -> posición del bloque en el buffer
      //    - MPI_COMM_WORLD  -> comunicador por el que se envía
      MPI_Pack( A( comienzo ), 1, MPI_BLOQUE, buf_envio, tam_buf_envio,
                &posicion, MPI_COMM_WORLD );

    }

  }

  // Creación del buffer de recepción
  int *buf

}
