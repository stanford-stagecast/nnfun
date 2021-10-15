#include "layer.hh"
#include "network.hh"
#include "timer.hh"
#include <Eigen/Dense>
#include <array>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  Network<2, 5, 3, 2, 1> nn;

  Matrix<float, 2, 5> inputs;
  inputs << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;

  nn.apply( inputs );

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  cout << "input:" << endl << inputs.format( CleanFmt ) << endl << endl;
  nn.print();
  return 0;
}
