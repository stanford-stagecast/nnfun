#include "exception.hh"
#include "network.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <utility>

#include <sys/resource.h>
#include <sys/time.h>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;

/* use squared error as loss function */
float loss_function( const float target, const float actual )
{
  return ( target - actual ) * ( target - actual );
}

/* partial derivative of loss with respect to neural network output */
float compute_pd_loss_wrt_output( const float target, const float actual )
{
  return -2 * ( target - actual );
}

/* actual function we want the neural network to learn */
float true_function( const float input )
{
  //return 3 * input + 1;
  return 1.0 / input;
}

float learning_rate = 0.001;

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 1>>();
  //nn->layer0.weights()( 0 ) = 1;
  //nn->layer0.biases()( 0 ) = 1;
  nn->layer0.initializeWeightsRandomly();

  for ( auto i = 0; i < 1000; i++ ) {
    /* step 1: construct a unique problem instance */
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    double x_value = rand();
    input( 0, 0 ) = x_value;
    ground_truth_output( 0, 0 ) = true_function( x_value );

    cout << "problem instance: " << input( 0, 0 ) << " => " << ground_truth_output( 0, 0 ) << "\n";

    /* step 2: forward propagate and calculate loss function */
    nn->apply( input );

    cout << "NN maps " << input( 0, 0 ) << " => " << nn->output()( 0, 0 ) << "\n";

    cout << "loss when " << ground_truth_output( 0, 0 ) << " desired, " << nn->output()( 0, 0 )
         << " produced = " << loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) ) << "\n";

    /* step 3: backpropagate error */
    nn->computeDeltas();
    nn->evaluateGradients( input );

    const float pd_loss_wrt_output
      = compute_pd_loss_wrt_output( ground_truth_output( 0, 0 ), nn->output()( 0, 0 ) );

    auto temp_learning_rate_one = 4.0 / 3 * learning_rate;
    auto temp_learning_rate_two = 2.0 / 3 * learning_rate;

    /* loss if not modifying weight or biase */
    auto current_weights = nn->layer0.weights()( 0 );
    auto current_biases = nn->layer0.biases()( 0 );

    auto current_loss = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    nn->layer0.weights()( 0 ) -= temp_learning_rate_one * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 );
    nn->layer0.biases()( 0 ) -= temp_learning_rate_one * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 );
    nn->apply( input );

    /* loss if decrementing eta */
    auto loss_learning_rate_one = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    nn->layer0.weights()( 0 )
      = current_weights - temp_learning_rate_two * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 );
    nn->layer0.biases()( 0 )
      = current_biases - temp_learning_rate_two * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 );
    nn->apply( input );

    /* loss if incrementing eta */
    auto loss_learning_rate_two = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    auto min_loss = min( min( loss_learning_rate_one, loss_learning_rate_two ), current_loss );
    cout << loss_learning_rate_one << " " << loss_learning_rate_two << " " << current_loss << endl;
    if ( min_loss == current_loss ) {
      learning_rate *= 2.0 / 3;
      nn->layer0.weights()( 0 ) = current_weights;
      nn->layer0.biases()( 0 ) = current_biases;
    } else if ( min_loss == loss_learning_rate_one ) {
      learning_rate *= 4.0 / 3;
      nn->layer0.weights()( 0 )
        = current_weights - temp_learning_rate_one * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 );
      nn->layer0.biases()( 0 )
        = current_biases - temp_learning_rate_one * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 );
      learning_rate *= 4.0 / 3;
    } else {
      nn->layer0.weights()( 0 )
        = current_weights - temp_learning_rate_two * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 );
      nn->layer0.biases()( 0 )
        = current_biases - temp_learning_rate_two * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 );
    }

    cout << "weight: " << nn->layer0.weights()( 0 ) << ", biase: " << nn->layer0.biases()( 0 ) << endl << endl;
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
