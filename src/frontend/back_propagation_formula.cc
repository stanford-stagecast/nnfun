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

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  // srand( Timer::timestamp_ns() );
  srand( 0 );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 4, 4, 2, 1>>();
  nn->initializeWeightsRandomly();

  srand( 10 );
  /* initialize inputs */
  Matrix<float, batch_size, input_size> input = Matrix<float, batch_size, input_size>::Random();

  /* forward prop */
  nn->apply( input );

  /* back prop */
  nn->computeDeltas();
  nn->evaluateGradients( input );

  /* print */
  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
  cout << "input:" << endl << input.format( CleanFmt ) << endl << endl;
  nn->print();

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
      float gradient = nn->getEvaluatedGradient( layerNum, paramNum );
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
    cout << endl;
    cout << endl << endl;
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