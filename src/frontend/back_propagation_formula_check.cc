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
// max allowable absolute difference in numerical and backprop gradients
constexpr double compare_epsilon = 1e-3;
// max allowable percentage difference in numerical and backprop gradients
constexpr double percentage_error_epsilon = 1e-3;

void program_body( const double epsilon )
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
  double maxDiff = 0;
  double maxPercentageError = 0.0;
  bool errorsOverThreshold = false;

  unsigned int numLayers = nn->getNumLayers();
  for ( unsigned int layerNum = 0; layerNum < numLayers; layerNum++ ) {
    unsigned int numParams = nn->getNumParams( layerNum );
    vector<double> gradients( numParams, 0 );
    cout << layerNum << endl;
    for ( unsigned int paramNum = 0; paramNum < numParams; paramNum++ ) {
      double formulaGradient = nn->getEvaluatedGradient( layerNum, paramNum );
      double numericalGradient = nn->calculateNumericalGradient( input, layerNum, paramNum, epsilon );
      double diff = abs( formulaGradient - numericalGradient );
      double percentageError = diff / max( abs( formulaGradient ), abs( numericalGradient ) );
      maxDiff = max( diff, maxDiff );
      maxPercentageError = max( percentageError, maxPercentageError );
      if ( ( ( diff > compare_epsilon ) and ( percentageError > percentage_error_epsilon ) ) ) {
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
  cout << "Params: grad_epsilon " << epsilon << ", compare_epsilon " << compare_epsilon
       << ", percentage_error_epsilon " << percentage_error_epsilon << endl;
  cout << "maxDiff: " << maxDiff << endl << "maxPercentageError: " << maxPercentageError << endl << endl;
  if ( errorsOverThreshold ) {
    cout << "test failure" << endl;
  }
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      cerr << "Usage: " << argv[0] << " EPSILON\n";
      return EXIT_FAILURE;
    }

    program_body( stof( argv[1] ) );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
