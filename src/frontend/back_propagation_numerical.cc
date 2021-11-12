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
constexpr size_t input_size = 3;

void program_body( const unsigned int num_iterations, const float epsilon )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  // srand( Timer::timestamp_ns() );
  srand( 0 );

  /* construct neural network on heap */
  auto nn = make_unique<Network<batch_size, input_size, 4, 4, 2, 1>>();
  nn->initializeWeightsRandomly();

  srand( 10 );
  /* initialize inputs */
  Matrix<float, batch_size, input_size> input = Matrix<float, batch_size, input_size>::Random();

  nn->apply( input );
  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  cout << "input:" << endl << input.format( CleanFmt ) << endl << endl;
  nn->print();

  vector<unsigned int> gradientLayerNums;
  vector<unsigned int> gradientWeightNums;

  unsigned int numLayers = nn->getNumLayers();

  for ( unsigned int layerNum = 0; layerNum < numLayers; layerNum++ ) {
    cout << "Layer " << layerNum << "\n";
    const unsigned int input_size_ = nn->getLayerInputSize( layerNum );
    const unsigned int output_size_ = nn->getLayerOutputSize( layerNum );
    cout << " input size: " << input_size_ << " -> "
         << "output_size: " << output_size_ << endl
         << endl;

    unsigned int numParams = nn->getNumParams( layerNum );
    vector<float> gradients( numParams, 0 );

    for ( unsigned int paramNum = 0; paramNum < numParams; paramNum++ ) {
      float gradient = nn->calculateNumericalGradient( input, layerNum, paramNum, epsilon );
      gradients[paramNum] = gradient;
    }

    cout << "  weightGradients ";
    for ( unsigned int paramNum = 0; paramNum < numParams; paramNum++ ) {
      if ( paramNum % output_size_ == 0 )
        cout << endl << "   ";
      if ( paramNum == input_size_ * output_size_ ) {
        cout << endl;
        cout << "  biasGradients" << endl << "   ";
      }
      cout << gradients[paramNum] << " ";
    }
    cout << endl << endl << endl;
  }

  // Code to generate random no. of such gradients
  (void)num_iterations;
  // for ( unsigned int i = 0; i < num_iterations; i++ ) {
  //   unsigned int layerNum = rand() % numLayers;
  //   gradientLayerNums.emplace_back( layerNum );
  //   unsigned int numParams = nn->getNumParams( layerNum );
  //   gradientWeightNums.emplace_back( rand() % numParams );
  // }

  // for ( unsigned int i = 0; i < num_iterations; i++ ) {
  //   float gradient = nn->calculateNumericalGradient( input, gradientLayerNums[i], gradientWeightNums[i], epsilon
  //   ); cout << "Iteration " << i << ":\n"
  //        << "  Layer " << gradientLayerNums[i] << "\n"
  //        << "  Weight " << gradientWeightNums[i] << "\n"
  //        << "  Gradient " << gradient << "\n\n\n\n";
  // }
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
