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

int main( int argc, char **argv ) {

  MPI_Init( &argc, & argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &id );

  switch( argc ) {
    case 3:
      NCIUDADES = atoi( argv[1] );
      break;
    default:
      cerr << "La sintaxis es: " << argv[0] << " <tamaño> <archivo>" << endl;
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

  U = INFINITO;       // Inicializamos la cuta superior a un valor muy grande
  InicNodo( &nodo );  // Inizializamos la estructura nodo

  if( id == 0 ) {
    LeerMatriz( argv[2], tsp0 );    // Leemos la matriz del fichero de entrada
    if( activo )
      pila.pop( nodo );
  }

  // Hacemos un Broadcast de la matriz, pues para que todos los procesos
  // funcionen bien necesitan conocer toda la matriz
  MPI_Bcast( &tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD );

  activo = !Inconsistente(tsp0);

  MPI_Barrier( MPI_COMM_WORLD );
  double tinit = MPI::Wtime();

  if( id != 0 ) {
    Equilibrado_Carga( pila, activo, id );
    if( activo )
      pila.pop( nodo );
  }

  while( activo ) {     // CICLO DEL BRANCH&BOUND

    //cout << "Proceso #" << id << " - it #" << iteraciones << endl;

    Ramifica( &nodo, &lnodo, &rnodo, tsp0 );
    nueva_U = false;

    if( Solucion( &rnodo ) ) {
      if( rnodo.ci() < U ) {
        U = rnodo.ci();        // Actualiza c.s.
        nueva_U = true;
        CopiaNodo( &rnodo, &solucion );
      }
    } else {
      if( rnodo.ci() < U ) {
        if( !pila.push( rnodo ) ) {
          liberarMatriz( tsp0 );
          exit(1);
        }
      }
    }

    if( Solucion( &lnodo ) ) {
      if( lnodo.ci() < U ) {
        U = lnodo.ci();        // Actualiza c.s.
        nueva_U = true;
        CopiaNodo( &lnodo, &solucion );
      }
    } else {
      if( lnodo.ci() < U ) {
        if( !pila.push( lnodo ) ) {
          liberarMatriz( tsp0 );
          exit(1);
        }
      }
    }

    //Difusion_Cota_Superior( &U );

    if( nueva_U )
      pila.acotar(U);

    Equilibrado_Carga( pila, activo, id );

    if( activo )
      pila.pop( nodo );

    iteraciones++;

    cout << "Proceso " << id << "\tIteración " << iteraciones << "\tSolución " << solucion.ci() << endl;

  }

  MPI_Barrier( MPI_COMM_WORLD );
  double tfin = MPI::Wtime();

  cout << "Proceso #" << id << " - número de iteraciones = " << iteraciones
       << endl << endl;

  MPI_Barrier( MPI_COMM_WORLD );

  printf( "Solución: \n" );
  EscribeNodo( &solucion );

  if( id == 0 ) {
    cout << "Tiempo gastado = " << tfin - tinit  << endl;
  }

  MPI_Finalize();

}
