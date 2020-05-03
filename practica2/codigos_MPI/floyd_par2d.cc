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
  int tam = nverts / raiz_P;    // Número de elementos por fila del bloque
  int bsize1d = nverts / size;  // Número de elementos totales del bloque
  int bsize2d = tam * tam;

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

      int comienzo = ( columna_P * tam ) + ( fila_P * bsize2d * raiz_P );

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
  int *buf_recep = new int[bsize2d];
  //int tam_buf_recep = sizeof( int ) * bsize1d;
  int tam_buf_recep = sizeof( int ) * bsize2d;

  // Distribución de la matriz entre los procesos, donde cada argumento indica:
  //    - buf_envio       -> datos sobre los que se aplica el scatter
  //    - tam_buf_recep   -> número de datos que llegan a cada proceso
  //    - MPI_PACKED      -> tipo de dato enviado
  //    - buf_recep       -> donde se almacenan los datos recibidos
  //    - tam * tam       -> número de datos que recibe cada proceso
  //    - MPI_INT         -> tipo de dato a recibir
  //    - 0               -> identificador del proceso que reparte los datos
  //    - MPI_COMM_WORLD  -> comunicador por el que se envia
  MPI_Scatter( buf_envio, tam_buf_recep, MPI_PACKED, buf_recep, bsize2d,
               MPI_INT, 0, MPI_COMM_WORLD );

  // Comprobación de la distribución -> OK
  // for( int i = 0; i < tam*tam; i++ )
  //   cout << "P"<<rank<<"["<<i<<"]="<<buf_recep[i]<<endl;

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 2                                            //
  //    CREACIÓN DE LOS COMUNICADORES VERTICAL Y          //
  //    HORIZONTAL                                        //
  //                                                      //
  //////////////////////////////////////////////////////////

  // Obtenemos los colores para cada comunicador
  int color_horizontal = rank / raiz_P;
  int color_vertical   = rank % raiz_P;

  // Creamos los comunicadores
  MPI_Comm comm_vertical, comm_horizontal;

  MPI_Comm_split( MPI_COMM_WORLD, color_horizontal, rank, &comm_horizontal );
  MPI_Comm_split( MPI_COMM_WORLD, color_vertical, rank, &comm_vertical );

  // Obtenemos los identificadores de cada proceso en cada comunicador
  int rank_horizontal, rank_vertical;

  MPI_Comm_rank( comm_horizontal, &rank_horizontal );
  MPI_Comm_rank( comm_vertical, &rank_vertical );

  // Comprobación de los comunicadores -> OK
  // if( rank == 0 )
  //   cout << endl << endl;
  //
  // cout << "Soy el Proceso " << rank << ", en el comunicador horizontal soy el " <<
  //         rank_horizontal << "º, y en el vertical, el " << rank_vertical << "º" << endl;

  //////////////////////////////////////////////////////////
  //                                                      //
  //    FASE 3                                            //
  //    IMPLEMENTACIÓN DEL ALGORITMO DE FLOYD_PAR2D       //
  //                                                      //
  //////////////////////////////////////////////////////////

  // Calculamos los i locales
  const int local_i_start = 0;
  const int local_i_end   = tam;

  // Y el i global de cada proceso
  const int global_i_start = ( rank / raiz_P ) * tam;
  const int global_i_end   = global_i_start + tam;

  // Y el j global de cada proceso
  const int global_j_start = ( rank % raiz_P ) * tam;
  const int global_j_end   = global_j_start + tam;

  // Vectores para almacenar las filas y columnas k
  int *fila_k    = new int[nverts];
  int *columna_k = new int[nverts];
  //int fila_k[tam];
  //int columna_k[nverts];
  int *fila_tmp, *columna_tmp;

  // Sincronizamos las hebras y tomamos la medida de tiempo inicial
  MPI_Barrier( MPI_COMM_WORLD );
  double tini = MPI_Wtime();

  // Bucle principal del algoritmo
  for( int k = 0; k < nverts; k++ ) {
    MPI_Barrier( MPI_COMM_WORLD );

    // Calculamos la fila y la columna k
    int row_k_process = k / tam;
    int col_k_process = row_k_process;

    // Si este proceso contiene parte de la fila k
    if( global_i_start <= k && k < global_i_end ) {
      const int local_k = k % tam;
      fila_tmp = fila_k;
      fila_k = &( buf_recep[local_k * tam] );
      // for( int i = 0; i < tam; i++ )
      //   cout << "P"<<rank<<"F"<<k<<"["<<i<<"]="<<fila_k[i]<<endl;
      // for( int i = 0; i < tam; i++ ) {
      //   int local_index = i * 1 + local_k;
      //   fila_k[i] = buf_recep[local_index];
      // }
    }

    // Broadcast de la fila k, donde cada parámetro indica:
    //    - fila_k            -> buffer de entrada y salida
    //    - tam               -> número de elementos que se envían
    //    - MPI_INT           -> tipo de dato que se envía
    //    - rok_k_process     -> identificador del comunicador que envía
    //    - comm_horizontal   -> comunicador por el que se envía
    MPI_Bcast( fila_k, tam, MPI_INT, row_k_process, comm_vertical );

    // Si este proceso contiene parte de la columna k
    if( global_j_start <= k && k < global_j_end ) {
      int local_k = k % tam;
      columna_tmp = columna_k;

      for( int i = 0; i < tam; i++ ) {
        columna_k[i] = buf_recep[local_k];
        local_k += tam;
        // cout << "P"<<rank<<"C"<<k<<"["<<i<<"]="<<columna_k[i]<<endl;
      }
    }

    // Broadcast de la columna k
    MPI_Bcast( columna_k, tam, MPI_INT, col_k_process, comm_horizontal );

    // Actualizamos la submatriz de cada proceso
    for( int i = local_i_start; i < local_i_end; i++ ) {

      int global_i = i + rank_vertical * tam;

      for( int j = local_i_start; j < local_i_end; j++ ) {

        int global_j = j + rank_horizontal * tam;

        if( global_i != global_j && global_i != k && global_j != k ) {
          int local_ij = i * tam + j;
          int suma_ikj = columna_k[i] + fila_k[j];
          buf_recep[local_ij] = min( buf_recep[local_ij], suma_ikj );
        }

      }

    }

  }

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
  MPI_Gather( buf_recep, bsize2d, MPI_INT, buf_envio, tam_buf_recep,
              MPI_PACKED, 0, MPI_COMM_WORLD );

  // Sincronizamos las hebras y tomamos la medida de tiempo final
  MPI_Barrier( MPI_COMM_WORLD );
  double tfin = MPI_Wtime();

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

    cout << endl << "Tiempo gastado: " << tfin - tini << " segundos." << endl << endl;

  }

  // Finalización de MPI
  MPI_Finalize();

}
