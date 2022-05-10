#include "exception.hh"
#include "network.hh"
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

float learning_rate = 0.001;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
constexpr size_t num_iterations = 10;
vector<unsigned int> layers { input_size, 10, 10, output_size };
// TODO do not know how to put the vector into nn

void program_body( Matrix<float, batch_size, input_size> input,
                   Matrix<float, batch_size, output_size> ground_truth_output )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 10, 10, output_size>>();

  nn->initializeWeightsRandomly();

  for ( int iter = 0; iter < (int)num_iterations; iter++ ) {
    nn->apply( input );

    nn->computeDeltas();
    nn->evaluateGradients( input );

    const float pd_loss_wrt_output
      = compute_pd_loss_wrt_output( ground_truth_output( 0, 0 ), nn->output()( 0, 0 ) );
    //float loss = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    for ( auto i = 0; i < (int)nn->getNumLayers(); i++ ) {
      //cout << "i: " << i << endl;

      for ( auto j = 0; j < (int)nn->getNumParams( i ); j++ ) {
        nn->modifyParam(
          (unsigned int)i, j, learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
      }
    }
  }
  input(0,0) = 3.0;
  nn->apply(input);
  cout << "result: " << nn->output()(0,0) << endl;
  nn->print();
}

int main( int argc, char*[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    double x_value = 2.0;
    input( 0, 0 ) = x_value;
    ground_truth_output( 0, 0 ) = true_function( x_value );
    program_body( input, ground_truth_output );
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    return EXIT_FAILURE;
  }
}
