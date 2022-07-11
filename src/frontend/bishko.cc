#include "neuralnetwork.hh"

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>

#include "timer.hh"

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 3;
constexpr size_t layer_size = 16;

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size, layer_size, output_size>>();
  nn->initialize(0.0000001);

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;

  /* TRAINING */
  int iterations = 50000;
  for (int i = 0; i < iterations; i++) {
    /* training function: y = 2x + 1, x = [1, 100] */
    input(0,0) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 99 + 1;
    ground_truth_output(0,0) = 2*input(0,0) + 1;
    nn->apply(input);
    nn->gradient_descent(input, ground_truth_output, true);

    if (i % (iterations / 100) == 0) {
        cout << (float)i/iterations * 100 << " Percent done training..." << endl;
    }
  }

  /* TESTING */
  float tot_diff = 0;
  for (int ipt = 1; ipt < 101; ipt++) {
    input(0,0) = ipt;
    nn->apply(input);
    float diff = abs((2 * ipt + 1) - nn->get_output()(0,0));
    cout << 2*ipt + 1 << " -> " << nn->get_output()(0,0) << " difference " << diff << endl;
    tot_diff += diff;
  }
  cout << "Average difference: " << tot_diff / 100 << endl;
  nn->printLayerOutput(0);
  nn->printLayerOutput(1);
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
    return EXIT_FAILURE;
  }
}