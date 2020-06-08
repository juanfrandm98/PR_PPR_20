/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/*                  Juan Francisco Díaz Moreno                          */
/*                  Junio de 2020                                       */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;
int id, size;
bool token_presente;
int estado;
int color;
MPI_Comm comunicadorCarga;
MPI_Comm comunicadorCota;
bool pendiente_retorno_cs;

int main( int argc, char **argv ) {

  MPI_Init( &argc, & argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &id );

  bool ejecucion_con_difusion;
  int eleccion;

  switch( argc ) {
    case 4:
      NCIUDADES = atoi( argv[1] );
      eleccion = atoi( argv[3] );

      if( eleccion == 0 )
        ejecucion_con_difusion = true;
      else
        ejecucion_con_difusion = false;

      break;
    default:
      cerr << "La sintaxis es: " << argv[0] << " <tamaño> <archivo> "
           << "<difusion(0 - sí, 1 - no)>" << endl;
      exit(1);
      break;
  }

  int **tsp0 = reservarMatrizCuadrada( NCIUDADES );
  tNodo nodo,     // Nodo a explorar
        lnodo,    // Hijo izquierdo
        rnodo,    // Hijo derecho
        solucion; // Mejor solución

  bool activo = true,    // Condición de fin
       nueva_U;   // Hay nuevo valor de c.s.

  int U;          // Valor de c.s.
  int iteraciones = 0;

  tPila pila;     // Pila de nodos a explorar

  // Variables añadidas para la detección de fin
  estado = 0;     // Inicialmente, los procesos están en estado activo
  color = 0;      // Inicialmente, los procesos son de color blanco

  // Variable añadida para la difusión de cota
  pendiente_retorno_cs = false;

  U = INFINITO;       // Inicializamos la cuta superior a un valor muy grande
  InicNodo( &nodo );  // Inizializamos la estructura nodo

  if( id == 0 ) {
    LeerMatriz( argv[2], tsp0 );    // Leemos la matriz del fichero de entrada
    cout << endl;
  }

  // Hacemos un Broadcast de la matriz, pues para que todos los procesos
  // funcionen bien necesitan conocer toda la matriz
  MPI_Bcast( &tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD );

  // Creamos los comunicadores
  MPI_Comm_split( MPI_COMM_WORLD, 0, id, &comunicadorCarga );
  MPI_Comm_split( MPI_COMM_WORLD, 1, id, &comunicadorCota );

  // El proceso 0 es el que comienza teniendo el token
  if( id == 0 )
    token_presente = true;
  else
    token_presente = false;

  activo = !Inconsistente(tsp0);

  MPI_Barrier( MPI_COMM_WORLD );
  double tinit = MPI::Wtime();

  if( id != 0 ) {
    Equilibrado_Carga( pila, activo, id, solucion );
    if( activo )
      pila.pop( nodo );
  }

  while( activo ) {     // CICLO DEL BRANCH&BOUND

    //cout << "Proceso #" << id << " - it #" << iteraciones << endl;
    //cout << "Proceso #" << id << ": inicio nueva iteración con " << pila.tamanio() << " en pila" << endl;

    Ramifica( &nodo, &lnodo, &rnodo, tsp0 );
    nueva_U = false;

    //cout << "Proceso #" << id << ": nodo a expandir de valor " << nodo.ci() << endl;
    //cout << "Proceso #" << id << ": rnodo.ci()=" << rnodo.ci() << "\tSolucion(&rnodo)=" << Solucion( &rnodo ) << endl;
    //cout << "Proceso #" << id << ": lnodo.ci()=" << lnodo.ci() << "\tSolucion(&lnodo)=" << Solucion( &lnodo ) << endl;

    if( Solucion( &rnodo ) ) {
      //cout << "Proceso #" << id << ": rnodo es solución - U=" << U << ", rnodo.ci=" << rnodo.ci() << endl;
      if( rnodo.ci() < U ) {
        //cout << "Proceso #" << id << ": modifico solución" << endl;
        U = rnodo.ci();        // Actualiza c.s.
        nueva_U = true;
        CopiaNodo( &rnodo, &solucion );
      }
    } else {
      //cout << "Proceso #" << id << ": rnodo no es solución, rnodo.ci()=" << rnodo.ci() << ", U=" << U << endl;
      if( rnodo.ci() < U ) {
        //cout << "Proceso #" << id << ": añado rnodo a la pila, su tamaño será " << pila.tamanio() + 1 << endl;
        if( !pila.push( rnodo ) ) {
					printf ("Error: pila agotada\n");
          liberarMatriz( tsp0 );
          exit(1);
        }
      }
    }

    if( Solucion( &lnodo ) ) {
      //cout << "Proceso #" << id << ": lnodo es solución - U=" << U << ", lnodo.ci=" << lnodo.ci() << endl;
      if( lnodo.ci() < U ) {
        //cout << "Proceso #" << id << ": modifico solución" << endl;
        U = lnodo.ci();        // Actualiza c.s.
        nueva_U = true;
        CopiaNodo( &lnodo, &solucion );
      }
    } else {
      //cout << "Proceso #" << id << ": lnodo no es solución, lnodo.ci()=" << lnodo.ci() << ", U=" << U << endl;
      if( lnodo.ci() < U ) {
        //cout << "Proceso #" << id << ": añado lnodo a la pila, su tamaño será " << pila.tamanio() + 1 << endl;
        if( !pila.push( lnodo ) ) {
					printf ("Error: pila agotada\n");
          liberarMatriz( tsp0 );
          exit(1);
        }
      }
    }

    //cout << "Proceso #" << id << ": iteración terminada - tamaño de la pila: " << pila.tamanio() << endl;

    if( ejecucion_con_difusion )
      Difusion_Cota_Superior( id, U, nueva_U );

    if( nueva_U )
      pila.acotar(U);

    Equilibrado_Carga( pila, activo, id, solucion );

    if( activo )
      pila.pop( nodo );

    iteraciones++;

    //cout << "Proceso #" << id << ": en mi " << iteraciones << "º iteración, mi cota es: " << U << endl;

  }

  MPI_Barrier( MPI_COMM_WORLD );
  double tfin = MPI::Wtime();

  cout << "Proceso #" << id << " - número de iteraciones = " << iteraciones << endl;

  MPI_Barrier( MPI_COMM_WORLD );

  if( id == 0 ) {
    cout << endl << "Solución: " << endl;
    EscribeNodo( &solucion );
    cout << "Tiempo gastado = " << tfin - tinit  << endl << endl;
  }

  MPI_Finalize();

}
