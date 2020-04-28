////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//        FLOYD_PAR2D                                                         //
//                                                                            //
//    PROGRAMA QUE EMPLEA EL ALGORITMO DE FLOYD DISTRIBUYENDO LA MATRIZ DE    //
//    ADYACENCIAS ENTRE VARIOS PROCESOS UTILIZANDO MPI_INT VERSIÓN QUE        //
//    DIVIDE LA MATRIZ USANDO SUBMATRICES 2D                                  //
//                                                                            //
//    JUAN FRANCISCO DÍAZ MORENO                                              //
//    PROGRAMACIÓN PARALELA                                                   //
//    MAYO DE 2020                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;

int main( int argc, char * argv[] ) {

  // Inicializamos MPI
  MPI_Init( &argc, &argv );

  Graph G;
  int nverts, rank, size;

  // Obtenemos el número de procesos y el ID de cada uno
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

    // Imprimimos la matriz inicial
    cout << "Grafo de entrada:" << endl;
    G.imprime();

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

  // Calculamos el número de elementos que tendrá cada submatriz (bloque)
  int raiz_P = sqrt( size );
  int tam = nverts / raiz_P;

  // Creación del buffer de envío para almacenar los datos empaquetados
  int *buf_envio = new int[nverts * nverts];
  int tam_buf_envio = sizeof( int ) * nverts * nverts;

  if( rank == 0 ) {

    // Obtenemos un vector a partir del grafo de entrada
    int *A = G.Get_Matrix();

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

      int comienzo = ( columna_P * tam ) + ( fila_P * tam * tam * raiz_P );

      // Se empaqueta el bloque i, donde cada argumento indica:
      //    - A(comienzo)     -> primer elemento a enviar
      //    - 1               -> número de elementos a enviar
      //    - MPI_BLOQUE      -> tipo de dato a enviar
      //    - buf_envio       -> buffer de envio
      //    - tam_buf_envio   -> tamaño del buffer de envio
      //    - &posicion       -> posición del bloque en el buffer
      //    - MPI_COMM_WORLD  -> comunicador por el que se envía
      MPI_Pack( &A[comienzo], 1, MPI_BLOQUE, buf_envio, tam_buf_envio,
                &posicion, MPI_COMM_WORLD );

    }

  }

  // Creación del buffer de recepción
  int *buf_recep = new int[tam * tam];
  int tam_buf_recep = sizeof( int ) * tam * tam;

  // Distribución de la matriz entre los procesos, donde cada argumento indica:
  //    - buf_envio       -> datos sobre los que se aplica el scatter
  //    - tam_buf_recep   -> número de datos que llegan a cada proceso
  //    - MPI_PACKED      -> tipo de dato enviado
  //    - buf_recep       -> donde se almacenan los datos recibidos
  //    - tam * tam       -> número de datos que recibe cada proceso
  //    - MPI_INT         -> tipo de dato a recibir
  //    - 0               -> identificador del proceso que reparte los datos
  //    - MPI_COMM_WORLD  -> comunicador por el que se envia
  MPI_Scatter( buf_envio, tam_buf_recep, MPI_PACKED, buf_recep, tam * tam,
               MPI_INT, 0, MPI_COMM_WORLD );

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 2                                            //
  //    CREACIÓN DE LOS COMUNICADORES VERTICAL Y          //
  //    HORIZONTAL                                        //
  //                                                      //
  //////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 1                                            //
  //    IMPLEMENTACIÓN DEL ALGORITMO DE FLOYD_PAR2D       //
  //                                                      //
  //////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 4                                            //
  //    RECOPILACIÓN FINAL DE LA MATRIZ RESULTADO         //
  //                                                      //
  //////////////////////////////////////////////////////////

  // Reunimos todas las submatrices en el proceso 0, donde cada argumento indica:
  //    - buf_recep       -> datos que envía cada proceso
  //    - tam * tam       -> número de datos que envía cada proceso
  //    - MPI_INT         -> tipo de dato que se envía
  //    - buf_envio       -> donde se almacenan todos los datos recibidos
  //    - tam_buf_recep   -> tamaño de los datos que se recibe de cada proceso
  //    - MPI_PACKED      -> Tipo de dato que se recibe
  //    - 0               -> identificador del proceso que recibe los envíos
  //    - MPI_COMM_WORLD  -> comunicador por el que se envían los datos
  MPI_Gather( buf_recep, tam * tam, MPI_INT, buf_envio, tam_buf_recep,
              MPI_PACKED, 0, MPI_COMM_WORLD );

  if( rank == 0 ) {

    // Creamos la matriz donde se almacenarán los datos
    int *B = new int[nverts * nverts];

    for( int i = 0, posicion = 0; i < size; i++ ) {

      // Cálculo de la posición donde se posicionará cada submatriz
      int fila_P = i / raiz_P;
      int columna_P = i % raiz_P;

      int comienzo = ( columna_P * tam ) + ( fila_P * tam * tam * raiz_P );

      // Desempaquetamos los datos enviados, donde cada argumento indica:
      //    - buf_envio       -> datos que se desempaquetan
      //    - tam_buf_envio   -> tamaño de los datos que se desempaquetan
      //    - posicion        -> posición del bloque en el buffer
      //    - B               -> donde se colocan los datos desempaquetados
      //    - 1               -> número de elementos que se desempaquetan
      //    - MPI_BLOQUE      -> elemento que se desempaqueta
      //    - MPI_COMM_WORLD  -> comunicador por el que se ha recibido
      MPI_Unpack( buf_envio, tam_buf_envio, &posicion, &B[comienzo], 1,
                  MPI_BLOQUE, MPI_COMM_WORLD );

    }

    // Mostramos el grafo resultado para hacer las comprobaciones
    cout << endl << endl << "Grafo de salida:" << endl;

    for( int i = 0; i < nverts; i++ ) {
      cout << "A[" << i << ",*]= ";
      for( int j = 0; j < nverts; j++ ) {
        int num = B[i * nverts + j];

        if( num == INF ) cout << "INF ";  // Para que el resultado salga con la
        else cout << num << " ";          // misma sintaxis que la matriz inicial,
      }                                   // mostramos 1000000 como INF
      cout << endl;
    }

  }

  // Finalización de MPI
  MPI_Finalize();

}
