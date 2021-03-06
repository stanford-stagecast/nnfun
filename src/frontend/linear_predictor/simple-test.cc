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
  return 3 * input + 1;
}

constexpr float learning_rate = 0.001;

void program_body()
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 1>>();
  nn->layer0.weights()( 0 ) = 1;
  nn->layer0.biases()( 0 ) = 1;

  double x_value = rand() % 30;
  // while ( true ) {
  for ( auto i = 0; i < 100; i++ ) {
    cout << "Weight = " << nn->layer0.weights()( 0 ) << ", bias = " << nn->layer0.biases()( 0 ) << "\n";

    /* step 1: construct a unique problem instance */
    Matrix<float, batch_size, input_size> input, ground_truth_output;

    input( 0, 0 ) = x_value;
    ground_truth_output( 0, 0 ) = true_function( x_value );

    // x_value++; /* use a different problem instance next time */
    x_value = rand() % 40;

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

    /* perturb weight */
    nn->layer0.weights()( 0 ) -= learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 );
    nn->layer0.biases()( 0 ) -= learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 );
    // nn -> layer0.weights()(0) -= learning_rate * pd_loss_wrt_output;
    // nn -> layer0.biases()(0) -= learning_rate * pd_loss_wrt_output * nn->layer0.biases()(0);

    cout << "\n";

    cout << "pd_loss_wrt_output = " << pd_loss_wrt_output << "\n";
    cout << "pd_output_wrt_weight = " << nn->getEvaluatedGradient( 0, 0 ) << "\n";
    cout << "pd_output_wrt_bias = " << nn->getEvaluatedGradient( 0, 1 ) << "\n";

    cout << "\n";

    cout << "perturb weight: " << -pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 0 ) << "\n";
    cout << "perturb bias: " << -pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 1 ) << "\n";

    cout << "\n";
  }

#if 0
  /* input and output */
  for ( int i = 0; i < 10; i++ ) {
    Matrix<float, batch_size, input_size> in, out;
    int v = rand() % 100 + 1;
    in( 0 ) = (float)v;
    out( 0 ) = 3 * v + 1 + (float)rand() / ( RAND_MAX );
    input.push_back( in );
    output.push_back( out );
  }
  for ( int round = 0; round < 100; round++ ) {
    for ( int i = 0; i < 10; i++ ) {
      nn->apply( input[i] );
      cout << "right after apply: " << input[i]( 0 ) << "  " << output[i]( 0 ) << endl;

      // cout << "output: " << nn->output() << "\n";

      // const float loss = ( 100 - nn->output()( 0 ) ) * ( 100 - nn->output()( 0 ) );
      const float loss = getLoss( output[i]( 0 ), nn->output()( 0 ) );

      nn->computeDeltas();
      nn->evaluateGradients( input[i] );

      // const float pd_loss_output = -2 * ( 100 - nn->output()( 0 ) );
      const float pd_loss_output = get_pd_loss_output( output[i]( 0 ), nn->output()( 0 ) );

      cout << "partial derivative of output wrt weight = " << nn->getEvaluatedGradient( 0, 0 ) << "\n";
      cout << "partial derivative of output wrt bias = " << nn->getEvaluatedGradient( 0, 1 ) << "\n";

      cout << "loss: " << loss << "\n";

      cout << "partial derivative of loss wrt output = " << pd_loss_output << "\n";

      cout << "partial derivative of loss wrt weight = " << pd_loss_output * nn->getEvaluatedGradient( 0, 0 )
           << "\n";
      cout << "partial derivative of loss wrt bias = " << pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) << "\n";
      cout << "weight = " << nn->layer0.weights()( 0 ) << endl;
      cout << "biase = " << nn->layer0.biases()( 0 ) << endl;

      const float learning_rate = .0001;

      nn->layer0.weights()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 0 ) );
      nn->layer0.biases()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) );
    }
  }

  // testing
  cout << "start testing results" << endl;
  Matrix<float, batch_size, input_size> test_input, test_output;
  for ( int i = 0; i < 10; i++ ) {
    int v = rand() % 100 + 1;
    test_input( 0 ) = (float)v;
    test_output( 0 ) = 3 * v + 1;
    cout << "input: " << test_input( 0 ) << ", output: " << test_output( 0 ) << endl;
    nn->apply( test_input );
    auto nnOutput = nn->output();
    cout << "nn output: " << nnOutput << endl;
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
