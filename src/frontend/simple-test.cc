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

  while ( true ) {
    nn->apply( input );

    cout << "output: " << nn->output() << "\n";

    const float loss = ( 100 - nn->output()( 0 ) ) * ( 100 - nn->output()( 0 ) );

    nn->computeDeltas();
    nn->evaluateGradients( input );

    const float pd_loss_output = -2 * ( 100 - nn->output()( 0 ) );

    cout << "partial derivative of output wrt weight = " << nn->getEvaluatedGradient( 0, 0 ) << "\n";
    cout << "partial derivative of output wrt bias = " << nn->getEvaluatedGradient( 0, 1 ) << "\n";

    cout << "loss: " << loss << "\n";

    cout << "partial derivative of loss wrt output = " << pd_loss_output << "\n";

    cout << "partial derivative of loss wrt weight = " << pd_loss_output * nn->getEvaluatedGradient( 0, 0 ) << "\n";
    cout << "partial derivative of loss wrt bias = " << pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) << "\n";

    const float learning_rate = .0001;

    nn->layer0.weights()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 0 ) );
    nn->layer0.biases()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) );
  }
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
