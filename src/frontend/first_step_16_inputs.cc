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
constexpr size_t input_size = 16;

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
float true_function( const Matrix<float, input_size, 1>& input )
{
  return input.sum();
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
  nn->layer0.initializeWeightsRandomly();

  /* test true function */
  //Matrix<float, input_size, 1> input(16);
  //input << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  //cout << true_function(input) << endl;
  for (auto i = 0; i < 15; i++) {
    /* step 1: construct a unique problem instance */
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    input << 1+i,2+i,3+i,4+i,5+i,6+i,7+i,8+i,9+i,10+i,11+i,12+i,13+i,14+i,15+i,16+i;
    ground_truth_output(0, 0) = true_function(input);

    cout << "problem instance: from " << input(0,0) << " to " << input(0,15) << " => " << ground_truth_output(0,0) << endl;

    /* step 2: forward propagate and calculate loss functiom */
    nn->apply( input );
    cout << "nn maps input: " << input(0,0) << " to " << input(0,15) << " => " << nn->output()(0,0) << endl;

    cout << "loss: " << loss_function (nn->output()(0,0), ground_truth_output(0,0)) << endl;

    /* step 3: backpropagate error */
    nn->computeDeltas();
    nn->evaluateGradients( input );

    const float pd_loss_wrt_output = compute_pd_loss_wrt_output( ground_truth_output(0,0), nn->output()(0,0) );

    // TODO: static eta -> dynamic eta
	learning_rate = 0.001;
    float loss = loss_function(nn->output()(0,0), ground_truth_output(0,0));

    for (int j = 0; j < 16; j++) {
      nn->layer0.weights()(0) -= learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient(0,j);
    }
    nn->layer0.biases()(0) -= learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient(0,16);

    cout << pd_loss_wrt_output << " " << loss << endl;
  }

#if 0
  for ( auto i = 0; i < 1000; i++ ) {
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
#endif
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