/**
  This file tries to predict 1/x given x
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

float true_function( const float input )
{
  return 1.0 / input;
}

/* compute input */
Matrix<float, batch_size, input_size> gen_time( float tempo, float offset )
{
  Matrix<float, batch_size, input_size> ret_mat;
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = tempo * i + offset;
  }
  return ret_mat;
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
  // nn->layer0.initializeWeightsRandomly();
  for ( int i = 0; i < 16; i++ ) {
    nn->layer0.weights()( i ) = 0;
  }
  nn->layer0.biases()( 0 ) = 0;

  //
  double x_value = ( rand() % 30 ) + 1;
  for ( auto i = 0; i < 100; i++ ) {
    /* test true function */
    /* step 1: construct a unique problem instance */
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    input( 0, 0 ) = x_value;
    ground_truth_output( 0, 0 ) = true_function( x_value );
    x_value = ( rand() % 40 ) + 1;

    /* step 2: forward propagate and calculate loss functiom */
    nn->apply( input );
    cout << "input: " << input << " => output: " << nn->output()( 0, 0 )
         << " => truth: " << ground_truth_output( 0, 0 ) << endl;

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

  for ( int i = 1; i < 100; i++ ) {
    Matrix<float, batch_size, input_size> input;
    input( 0, 0 ) = i * 1.0;
    nn->apply( input );
    // cout << "input: " << i << " output: " << 60.0 / nn->output()( 0, 0 ) << endl;
    cout << "input: " << i << " output: " << nn->output()( 0, 0 ) << " ground truth: " << true_function( i * 1.0 )
         << endl;
    // cout << i << endl;
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
