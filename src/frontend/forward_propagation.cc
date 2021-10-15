#include "network.hh"
#include "timer.hh"
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  srand( time( NULL ) );
  Network<2, 5, 3, 2, 1> nn;

  Matrix<float, 2, 5> inputs;
  inputs << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;

  uint64_t start = Timer::timestamp_ns();
  nn.apply( inputs );
  uint64_t end = Timer::timestamp_ns();
  cout << "TIME: " << end - start << endl;

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  cout << "input:" << endl << inputs.format( CleanFmt ) << endl << endl;
  nn.print();

  return 0;
}
