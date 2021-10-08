#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

#include "network.hh"

using namespace std;
using namespace Eigen;

void program_body()
{
  Matrix<float, 2, 1> input;
  input << 9, 2;

  Matrix<float, 3, 2> layer1;
  layer1 << 1, 2, 3, 4, 5, 6;

  const auto output = layer1 * input;

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

  cout << layer1.format( CleanFmt ) << "\n*\n"
       << input.format( CleanFmt ) << "\n=\n"
       << output.format( CleanFmt ) << "\n";
}

int main()
{
  try {
    ios::sync_with_stdio( false );
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
