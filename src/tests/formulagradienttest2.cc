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
constexpr float grad_epsilon = 1e-3;
constexpr float compare_epsilon = 1e-3;
constexpr float percentage_error_epsilon = 5 * 1e-2;

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  // srand( Timer::timestamp_ns() );
  srand( 0 );

  /* construct neural network on heap */
  auto nn = make_unique<Network<batch_size, input_size, 4, 64, 1>>();
  nn->initializeWeightsRandomly();

  srand( 10 );
  /* initialize inputs */
  Matrix<float, batch_size, input_size> input = Matrix<float, batch_size, input_size>::Random();

  nn->apply( input );
  nn->computeDeltas();
  nn->evaluateGradients( input );

  bool errorsOverThreshold = false;
  float maxDiff = 0.0;
  float maxPercentageError = 0.0;
  unsigned int numLayers = nn->getNumLayers();
  for ( unsigned int layerNum = 0; layerNum < numLayers; layerNum++ ) {
    unsigned int numParams = nn->getNumParams( layerNum );
    vector<float> gradients( numParams, 0 );
    for ( unsigned int paramNum = 0; paramNum < numParams; paramNum++ ) {
      float formulaGradient = nn->getEvaluatedGradient( layerNum, paramNum );
      float numericalGradient = nn->calculateNumericalGradient( input, layerNum, paramNum, grad_epsilon );
      float diff = abs( formulaGradient - numericalGradient );
      float percentageError = diff / max( abs( formulaGradient ), abs( numericalGradient ) );
      maxDiff = max( diff, maxDiff );
      maxPercentageError = max( percentageError, maxPercentageError );
      if ( ( diff > compare_epsilon ) and ( percentageError > percentage_error_epsilon ) ) {
        errorsOverThreshold = true;
        cout << "Error in Layer " << layerNum << ", Param " << paramNum << endl;
        cout << formulaGradient << " " << numericalGradient << endl;
        cout << "diff: " << diff << ", %diff: " << percentageError << endl;
        // cout << "maxDiff: " << maxDiff << ", maxPercentageError: " << maxPercentageError << endl;
        cout << endl;
      }
    }
  }
  cout << endl;
  cout << "Params: grad_epsilon " << grad_epsilon << ", compare_epsilon " << compare_epsilon
       << ", percentage_error_epsilon " << percentage_error_epsilon << endl;
  cout << "maxDiff: " << maxDiff << endl << "maxPercentageError: " << maxPercentageError << endl << endl;
  if ( errorsOverThreshold ) {
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
