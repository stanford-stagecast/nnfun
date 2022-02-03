#include "exception.hh"
#include "network.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <random>

#include <sys/resource.h>
#include <sys/time.h>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;

vector<Matrix<float, batch_size, input_size>> input {};
vector<Matrix<float, batch_size, input_size>> output{};

float getLoss(float expect, float nnOutput) {
  return (expect - nnOutput) * (expect - nnOutput);
}

float get_pd_loss_output(float expect, float nnOutput) {
  cout << "inside: " << expect << " " << nnOutput << endl;
  return -2 * (expect - nnOutput);
}

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
  
  /* input and output */
  for (int i = 0; i < 10; i++) {
    Matrix<float, batch_size, input_size> in, out;
    int v = rand() % 100 + 1;
    in(0) = (float)v;
    out(0) = 3 * v + 1 + (float)rand()/(RAND_MAX);
    input.push_back(in);
    output.push_back(out);
  }
  for (int round = 0; round < 100; round++) {
    for (int i = 0; i < 10; i++) {
      nn->apply( input[i] );
      cout << input[i](0) << "  " << output[i](0) << endl;

      //cout << "output: " << nn->output() << "\n";

      //const float loss = ( 100 - nn->output()( 0 ) ) * ( 100 - nn->output()( 0 ) );
      const float loss = getLoss(output[i](0), nn->output()(0));

      nn->computeDeltas();
      nn->evaluateGradients( input[i] );

      //const float pd_loss_output = -2 * ( 100 - nn->output()( 0 ) );
      const float pd_loss_output = get_pd_loss_output(output[i](0), nn->output()(0));

      cout << "partial derivative of output wrt weight = " << nn->getEvaluatedGradient( 0, 0 ) << "\n";
      cout << "partial derivative of output wrt bias = " << nn->getEvaluatedGradient( 0, 1 ) << "\n";

      cout << "loss: " << loss << "\n";

      cout << "partial derivative of loss wrt output = " << pd_loss_output << "\n";

      cout << "partial derivative of loss wrt weight = " << pd_loss_output * nn->getEvaluatedGradient( 0, 0 ) << "\n";
      cout << "partial derivative of loss wrt bias = " << pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) << "\n";
      cout << "weight = " << nn->layer0.weights()(0) << endl;
      cout << "biase = " << nn->layer0.biases()(0) << endl;

      const float learning_rate = .0001;

      nn->layer0.weights()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 0 ) );
      nn->layer0.biases()( 0 ) *= ( 1 - learning_rate * pd_loss_output * nn->getEvaluatedGradient( 0, 1 ) );
    }
  }
 
  // testing
  cout << "start testing results" << endl;
  Matrix<float, batch_size, input_size> test_input, test_output;
  for (int i = 0; i < 10; i++) {
    int v = rand() % 100 + 1;
    test_input(0) = (float)v;
    test_output(0) = 3 * v + 1;
    cout << "input: " << test_input(0) << ", output: " << test_output(0) << endl;
    nn -> apply( test_input );
    auto nnOutput = nn -> output();
    cout << "nn output: " << nnOutput << endl;
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
