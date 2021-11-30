#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

void program_body()
{
  Matrix<double, 2, 1> input;
  input << 9, 2;

  Matrix<double, 3, 2> layer1;
  layer1 << 1, 2, 3, 4, 5, 6;

  auto output = layer1 * input;

  Matrix<double, 3, 1> expected_output;
  expected_output << 13, 35, 57;

  if ( output != expected_output ) {
    throw runtime_error( "test failure" );
  }
}

int main()
{
  try {
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
