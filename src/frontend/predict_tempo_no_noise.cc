/**
  This file creates a one-layer neural network to calculate the beat from 16
  inputs.
  
  We want range of tempo: 35 - 250 bpm //TODO
*/
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

float learning_rate = 0.00001;

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
  // Matrix<float, input_size, 1> input(16);
  // input << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  // cout << true_function(input) << endl;
  // for ( auto i = 0; i < 1000; i++ ) {
  int i = 0;
  while ( true ) {
    i++;
    /* step 1: construct a unique problem instance */
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    // input << 1 + i, 2 + i, 3 + i, 4 + i, 5 + i, 6 + i, 7 + i, 8 + i, 9 + i, 10 + i, 11 + i, 12 + i, 13 + i, 14 +
    // i, 15 + i, 16 + i;
    for ( int j = 0; j < 16; j++ ) {
      input( j ) = rand();
    }
    ground_truth_output( 0, 0 ) = true_function( input );

    cout << "problem instance: from " << input( 0, 0 ) << " to " << input( 0, 15 ) << " => "
         << ground_truth_output( 0, 0 ) << endl;

    /* step 2: forward propagate and calculate loss functiom */
    nn->apply( input );
    cout << "nn maps input: " << input( 0, 0 ) << " to " << input( 0, 15 ) << " => " << nn->output()( 0, 0 )
         << endl;

    cout << "loss: " << loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) ) << endl;

    /* step 3: backpropagate error */
    nn->computeDeltas();
    nn->evaluateGradients( input );

    const float pd_loss_wrt_output
      = compute_pd_loss_wrt_output( ground_truth_output( 0, 0 ), nn->output()( 0, 0 ) );

    // TODO: static eta -> dynamic eta
    auto four_third_lr = 4.0 / 3 * learning_rate;
    auto two_third_lr = 2.0 / 3 * learning_rate;

    /* calculate three loss */
    float current_loss = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );
    Matrix<float, input_size, 1> current_weights;
    for ( int j = 0; j < 16; j++ ) {
      current_weights( j ) = nn->layer0.weights()( j );
    }
    auto current_biase = nn->layer0.biases()( 0 );

    /* loss for 4/3 eta */
    for ( int j = 0; j < 16; j++ ) {
      nn->layer0.weights()( j ) -= four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
    }
    nn->layer0.biases()( 0 ) -= four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
    nn->apply( input );
    auto loss_four_third_lr = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    /* loss for 2/3 eta */
    for ( int j = 0; j < 16; j++ ) {
      nn->layer0.weights()( j )
        = current_weights( j ) - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
    }
    nn->layer0.biases()( 0 )
      = current_biase - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
    nn->apply( input );
    auto loss_two_third_lr = loss_function( nn->output()( 0, 0 ), ground_truth_output( 0, 0 ) );

    cout << current_loss << " " << loss_four_third_lr << " " << loss_two_third_lr << endl;
    auto min_loss = min( min( current_loss, loss_four_third_lr ), loss_two_third_lr );
    if ( min_loss == current_loss ) {
      learning_rate *= 2.0 / 3;
      for ( int j = 0; j < 16; j++ ) {
        nn->layer0.weights()( j ) = current_weights( j );
      }
      nn->layer0.biases()( 0 ) = current_biase;
    } else if ( min_loss == loss_four_third_lr ) {
	  learning_rate *= 4.0 / 3;
      for ( int j = 0; j < 16; j++ ) {
        nn->layer0.weights()( j )
          = current_weights( j ) - four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
      }
      nn->layer0.biases()( 0 )
        = current_biase - four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
    } else {
      for ( int j = 0; j < 16; j++ ) {
        nn->layer0.weights()( j )
          = current_weights( j ) - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
      }
      nn->layer0.biases()( 0 )
        = current_biase - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
    }

    cout << "weights: " << nn->layer0.weights() << endl;
    cout << "biase: " << nn->layer0.biases()( 0 ) << endl;
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
