#include "network.hh"
#include "timer.hh"
#include <Eigen/Dense>
#include <iostream>
#include <utility>

using namespace std;
using namespace Eigen;

int main()
{
  srand( time( NULL ) );
  auto nn
    = make_unique<Network<1, 128, 256, 512, 1024, 2048, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1>>();

  Matrix<float, 1, 128> inputs {};

  uint64_t start = Timer::timestamp_ns();
  nn->apply( inputs );
  uint64_t end = Timer::timestamp_ns();
  cout << "TIME: " << end - start << endl;

  /*
  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  cout << "input:" << endl << inputs.format( CleanFmt ) << endl << endl;
  nn.print();
  */

  return 0;
}
