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
constexpr size_t input_size = 64;
// epsilon for computation of numerical gradients
constexpr double grad_epsilon = 1e-5;
// max allowable absolute difference in numerical and backprop gradients
constexpr double compare_epsilon = 1e-5;
// max allowable percentage difference in numerical and backprop gradients
constexpr double percentage_error_epsilon = 1e-3;

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  // srand( Timer::timestamp_ns() );
  srand( 0 );

  /* construct neural network on heap */
  auto nn = make_unique<Network<double, batch_size, input_size, 64, 64, 64, 64, 64, 64, 64, 64, 1>>();
  nn->initializeWeightsRandomly();

  srand( 10 );
  /* initialize inputs */
  Matrix<double, batch_size, input_size> input = Matrix<double, batch_size, input_size>::Random();

  /* forward prop */
  nn->apply( input );

  /* back prop */
  nn->computeDeltas();
  nn->evaluateGradients( input );

  /* compare back prop results with numerical gradients */
  bool errorsOverThreshold = false;
  double maxDiff = 0.0;
  double maxPercentageError = 0.0;
  unsigned int numLayers = nn->getNumLayers();
  for ( unsigned int layerNum = 0; layerNum < numLayers; layerNum++ ) {
    unsigned int numParams = nn->getNumParams( layerNum );
    for ( unsigned int paramNum = 0; paramNum < numParams; paramNum++ ) {
      double formulaGradient = nn->getEvaluatedGradient( layerNum, paramNum );
      double numericalGradient = nn->calculateNumericalGradient( input, layerNum, paramNum, grad_epsilon );
      double diff = abs( formulaGradient - numericalGradient );
      double percentageError = diff / max( abs( formulaGradient ), abs( numericalGradient ) );
      maxDiff = max( diff, maxDiff );
      maxPercentageError = max( percentageError, maxPercentageError );
      if ( ( diff > compare_epsilon ) and ( percentageError > percentage_error_epsilon ) ) {
        errorsOverThreshold = true;
        cout << "Error in Layer " << layerNum << ", Param " << paramNum << endl;
        cout << formulaGradient << " " << numericalGradient << endl;
        cout << "diff: " << diff << ", %diff: " << percentageError << endl;
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
