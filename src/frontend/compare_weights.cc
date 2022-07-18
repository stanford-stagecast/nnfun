#include "neuralnetwork.hh"

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
constexpr size_t num_layers = 5;
constexpr size_t layer_size1 = 16;
constexpr size_t layer_size2 = 1;
constexpr size_t layer_size3 = 30;
constexpr size_t layer_size4 = 2560;

Matrix<float, batch_size, input_size> gen_time( float tempo, bool offset, bool noise )
{
  Matrix<float, batch_size, input_size> ret_mat; //empty matrix [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  float amt_offset = 0;
  if (offset) {
    float sbb  = (60.0 / tempo);   // seconds between beats
    amt_offset = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX )) * sbb;
  }
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = (60.0/tempo) * i + offset; //[0, 0.25, 0.5, 0.75, 1, 1.25, ...] (backward in time)
    if (noise) {
      float pct_noise = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX )) * 0.1 - 0.05;
      float amt_noise = pct_noise * (60.0/tempo);
      ret_mat(i) += amt_noise;
    }
    ret_mat(i) += amt_offset;
  }
  return ret_mat;
}

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size1, layer_size2, layer_size3, layer_size4, output_size>>();
  nn->initialize(0.0000001);
  string printed_weights_file = "/home/mirandan/nnfun/src/frontend/leaky_weights.txt";
  nn->init_params(printed_weights_file);
  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  // input = Matrix<float, batch_size, input_size>::Random();
  //input << 1;
  //ground_truth_output << 60.0 / 1;
  // nn->apply( input );
  // cout << nn->get_output()( 0, 0 ) << endl;
  // const uint64_t b_start = Timer::timestamp_ns();


 /******************** TRAINING ********************/
 /* bool offset = true;
  bool noise = true;
  int iterations = 2000000;
  for ( int i = 0; i < iterations; i++ ) {
      float tempo = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 210 + 30;
      input = gen_time( tempo, offset, noise );

      // Print Progress
      if(i % 1000 == 0) {
      cout << "training " << tempo << " bpm... (" << (float(i)/iterations)*100 << " percent done)" << endl;
      }

      // Train the neural network
      ground_truth_output(0,0) = tempo;
      nn->apply(input);
      nn->gradient_descent( input, ground_truth_output, true );
  }
  
  cout << endl;
  cout << "******* TRAINING COMPLETE :) *******" << endl;
  cout << endl;
  cout << "Testing neural network now..." << endl;
  cout << endl;

  // const uint64_t b_end = Timer::timestamp_ns();
  // cout << "timer: " << ( b_end - b_start ) << endl;

  // nn->print();
  */
 /******************** TESTING ********************/
 bool offset = true;
 bool noise = true;
  for (  int tempo = 30; tempo < 241; tempo++ ) {
    input = gen_time( tempo, offset, noise );
    // input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 1.75 + 0.25;
    // ground_truth_output( 0, 0 ) = 60.0 / input( 0, 0 );
    nn->apply_leaky(input);
    cout << tempo << " -> " << nn->get_output()( 0, 0 ) << endl;
  }
  cout << endl;
  cout << "Number of layers: " << num_layers << endl;
  cout << "Size of layer 1: " << layer_size1 << endl;
  cout << "With noise? " << noise << endl;
  cout << "With offset? " << offset << endl;
  //cout << "Number of Iterations: " << iterations << endl;
  cout << "Learning Rate: " << nn->get_current_learning_rate() << endl;

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