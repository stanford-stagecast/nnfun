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
constexpr size_t input_size = 16;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 4;
constexpr size_t layer_size = 448;

Matrix<float, batch_size, input_size> gen_time( float tempo, float offset, bool noise)
{
  Matrix<float, batch_size, input_size> ret_mat; //empty matrix [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = (60.0/tempo) * i + offset; //[0, 0.25, 0.5, 0.75, 1, 1.25, ...] (backward in time)
    if (noise) {
      float pct_noise = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX )) * 0.1 - 0.05;
      float amt_noise = pct_noise * (60.0/tempo);
      ret_mat(i) += amt_noise;
    }
  }
  return ret_mat;
}

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size, layer_size, layer_size, output_size>>();
  nn->initialize(0.0000001);

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;

  /* training */
  float offset = 0;
  int init_iterations = 500000;
  int second_iterations = 2500000;
  for ( int i = 0; i < (init_iterations + second_iterations); i++ ) {
    /* train on 30-60 range after 500000 iterations */
    float tempo;
    if (i > init_iterations) {
      tempo = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 30 + 30;
    } else {
    /* train on full range otherwise */
      tempo = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 210 + 30;
    }
    ground_truth_output(0,0) = tempo;
    input = gen_time(tempo, offset, true);
    nn->apply(input);
    nn->gradient_descent( input, ground_truth_output, true );
    if (i % ((init_iterations + second_iterations)/100) == 0) {
      cout << ((float)i/(init_iterations + second_iterations)) * 100 << " percent done training..." << endl;
    }
  }
  /* testing */
  float tot_diff = 0;
  float t_high = 240;
  float t_low = 30;
  for (  int tempo = 30; tempo < t_high + 1; tempo++ ) {
    input = gen_time( tempo, offset, true);
    nn->apply(input);
    float difference = abs(tempo - nn->get_output()(0,0));
    cout << tempo << " -> " << nn->get_output()( 0, 0 ) << " difference: " << difference << endl;
    tot_diff += difference;
  }
  cout << "Average Difference: " << tot_diff / (t_high - t_low) << endl;
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