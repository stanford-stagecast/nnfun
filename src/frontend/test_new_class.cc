#include "exception.hh"
#include "neuralnetwork.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include <sys/resource.h>

using namespace std;
using namespace Eigen;

float loss_function( const float target, const float actual )
{
  return ( target - actual ) * ( target - actual );
}

float compute_pd_loss_wrt_output( const float target, const float actual )
{
  return -2 * ( target - actual );
}

float true_function( const float input )
{
  return 1.0 / input;
}

constexpr size_t batch_size = 1;
constexpr size_t input_size = 16;
constexpr size_t output_size = 1;

void program_body()
{
  // num_layer = 2
  // batch_size = 1
  // input_size = 6
  // output_size = 1
  // 16 -> 16 -> 1
  // 2 1 16 1 16 1
  auto nn = make_unique<NeuralNetwork<float, 2, batch_size, input_size, output_size, 16, 1>>();
  nn->initialize();
  nn->print();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  input = Matrix<float, 1, 16>::Random();
  ground_truth_output << 40;
  nn->apply( input );
  cout << nn->get_output()( 0, 0 ) << endl;
  nn->gradient_descent( input, ground_truth_output, 0.0001 );

  nn->print();
}

int main( int argc, char*[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    return EXIT_FAILURE;
  }
}
