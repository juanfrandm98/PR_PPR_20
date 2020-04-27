#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

double calculaPI( long num_steps ) {

  double step;
  int i; double x, pi, sum = 0.0;

  step = 1.0 / (double) num_steps;

  for( i = 1; i <= num_steps; i++ ) {
    x = ( i - 0.5 ) * step;
    sum = sum + 4.0 / ( 1.0 + x * x );
  }

  pi = step * sum;

  return pi;

}

int main() {

  long num_steps;
  high_resolution_clock::time_point tantes, tdespues;
  duration<double> transcurrido;

  cout << "/////////////////////////////////////////////////" << endl;
  cout << "/// Calculo del número PI de forma secuencial ///" << endl;
  cout << "/////////////////////////////////////////////////" << endl;
  cout << endl << "Introduce el número de pasos: ";
  cin >> num_steps;

  tantes = high_resolution_clock::now();
  double pi = calculaPI( num_steps );
  tdespues = high_resolution_clock::now();

  transcurrido = duration_cast<duration<double>> ( tdespues - tantes );

  cout << endl << "Resultado: " << pi << "." << endl;
  cout << "El tiempo empleado es " << transcurrido.count() << " segundos." <<
  endl << endl;

}
