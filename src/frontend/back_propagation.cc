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
constexpr size_t input_size = 4;

void program_body( const unsigned int num_iterations, const float epsilon )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<batch_size, input_size, 5, 3, 2, 1>>();
  nn->initializeWeightsRandomly();

  /* initialize inputs */
  Matrix<float, batch_size, input_size> input = Matrix<float, batch_size, input_size>::Random();

  nn->apply( input );

  srand( 0 );

  vector<unsigned int> gradientLayerNums;
  vector<unsigned int> gradientWeightNums;

  unsigned int numLayers = nn->getNumLayers();

  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    unsigned int layerNum = rand() % numLayers;
    gradientLayerNums.emplace_back( layerNum );
    unsigned int numParams = nn->getNumParams( layerNum );
    gradientWeightNums.emplace_back( rand() % numParams );
  }

  for ( unsigned int i = 0; i < num_iterations; i++ ) {
    float gradient = nn->calculateNumericalGradient( input, gradientLayerNums[i], gradientWeightNums[i], epsilon );
    cout << "Iteration " << i << ":\n"
         << "  Layer " << gradientLayerNums[i] << "\n"
         << "  Weight " << gradientWeightNums[i] << "\n"
         << "  Gradient " << gradient << "\n\n\n\n";
  }

  // const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  // cout << "input:" << endl << inputs[num_iterations - 1].format( CleanFmt ) << endl << endl;
  // nn->print();
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 3 ) {
      cerr << "Usage: " << argv[0] << " NUM_ITERATIONS EPSILON\n";
      return EXIT_FAILURE;
    }

    program_body( stoi( argv[1] ), stof( argv[2] ) );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
