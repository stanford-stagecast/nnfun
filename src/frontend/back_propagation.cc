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
constexpr size_t input_size = 1024;

void program_body( const unsigned int num_iterations )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  // auto nn = make_unique<Network<float, batch_size, input_size, 4096, 1>>();
  // auto nn = make_unique<Network<float, batch_size, input_size, 2048, 2048, 1>>();
  auto nn = make_unique<Network<float, batch_size, input_size, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1>>();
  // auto nn = make_unique<Network<float, batch_size, input_size, 5, 3, 2, 1>>();
  nn->initializeWeightsRandomly();

  /* initialize inputs */
  vector<Matrix<float, batch_size, input_size>> inputs;
  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    inputs.emplace_back( Matrix<float, batch_size, input_size>::Random() );
  }

  /* run forward prop benchmark */
  const uint64_t f_start = Timer::timestamp_ns();
  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    /* forward prop */
    nn->apply( inputs[i] );
  }
  const uint64_t f_end = Timer::timestamp_ns();

  /* run backprop benchmark */
  const uint64_t b_start = Timer::timestamp_ns();
  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    /* forward prop */
    nn->apply( inputs[i] );
    /* back prop*/
    nn->computeDeltas();
    nn->evaluateGradients( inputs[i] );
  }

  const uint64_t b_end = Timer::timestamp_ns();

  cout << "Average back propagation time (over " << num_iterations << " iterations, batch size=" << batch_size
       << "): ";
  Timer::pp_ns( cout, ( ( b_end - b_start ) - ( f_end - f_start ) ) / float( num_iterations ) );
  cout << " per iteration\n";
  cout << "Back Propagation time / forward propagation time: "
       << ( ( b_end - b_start ) * 1.0 / ( f_end - f_start ) - 1 ) << " x\n";
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      cerr << "Usage: " << argv[0] << " NUM_ITERATIONS\n";
      return EXIT_FAILURE;
    }

    program_body( stoi( argv[1] ) );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
