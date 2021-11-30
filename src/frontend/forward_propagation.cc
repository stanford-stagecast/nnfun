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

constexpr size_t batch_size = 3;
constexpr size_t input_size = 1;

void program_body( const unsigned int num_iterations )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<batch_size, input_size, 1024, 1024, 1024, 1024, 1024, 1024, 1>>();
  // auto nn = make_unique<Network<batch_size, input_size, 5, 3, 2, 1>>();
  nn->initializeWeightsRandomly();

  /* initialize inputs */
  vector<Matrix<double, batch_size, input_size>> inputs;
  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    inputs.emplace_back( Matrix<double, batch_size, input_size>::Random() );
  }

  /* run benchmark */
  const uint64_t start = Timer::timestamp_ns();
  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    nn->apply( inputs[i] );
  }

  // const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  // cout << "input:" << endl << inputs[num_iterations-1].format( CleanFmt ) << endl << endl;
  // nn->print();

  const uint64_t end = Timer::timestamp_ns();

  cout << "Average runtime (over " << num_iterations << " iterations, batch size=" << batch_size << "): ";
  Timer::pp_ns( cout, ( end - start ) / double( num_iterations ) );
  cout << " per iteration\n";
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
