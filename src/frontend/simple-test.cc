#include "exception.hh"
#include "network.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <iostream>
#include <utility>

#include <sys/resource.h>
#include <sys/time.h>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 1>>();
  nn->layer0.weights()( 0 ) = 3;
  nn->layer0.biases()( 0 ) = 1;

  Matrix<float, 1, 1> input {};
  input( 0 ) = 2;

  nn->apply( input );

  cout << "output: " << nn->output() << "\n";

  nn->computeDeltas();
  nn->evaluateGradients( input );

  cout << "partial derivative of output wrt weight = " << nn->getEvaluatedGradient( 0, 0 ) << "\n";
  cout << "partial derivative of output wrt bias = " << nn->getEvaluatedGradient( 0, 1 ) << "\n";
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
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
