/**
 * File name: big_nn_1.cc
 * Last Update: July 12 2022
 * Description: This file uses a neural network (5 layers) that predicts
 *              tempo given 16 timestamps. It uses training data from a .txt 
 *              file (big_nn_1_weights.txt). The text file was created
 *              by [manually] concatenating the weights/biases data from
 *              two separately trained Neural Networks:
 *                  
 *                  NN1: 16 timestamps -> seconds/beat
 *                  NN2: seconds/beat -> tempo (bpm)
 *              
 *              Big Neural Network: 16 timestamps -> tempo (bpm)
 */

#include "neuralnetwork.hh"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include "timer.hh"

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t nn1_input_size = 16;
// constexpr size_t nn2_input_size = 1;
constexpr size_t output_size = 1;
// constexpr size_t nn1_num_layers = 2;
// constexpr size_t nn2_num_layers = 3;
constexpr size_t big_nn_num_layers = 5;
constexpr size_t big_nn_input_size = 16;

Matrix<float, batch_size, nn1_input_size> gen_time( float tempo, bool offset, bool noise )
{
  Matrix<float, batch_size, nn1_input_size> ret_mat; //empty matrix [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
  // The below commented-out code initialized and trained the two separate NN1 and NN2
  
  // // NN1: Toy Problem Solution (16->16->1)
  // auto nn1 = make_unique<
  //   NeuralNetwork<float, nn1_num_layers, batch_size, nn1_input_size, output_size, 16, output_size>>();
  // nn1->initialize(0.0000001);
  // // NN2: Solution to 60/x, no noise (1->480->480->1)
  // auto nn2 = make_unique<
  //   NeuralNetwork<float, nn2_num_layers, batch_size, nn2_input_size, output_size, 480, 480, output_size>>();
  // nn2->initialize(0.0000001);
  // Matrix<float, batch_size, nn2_input_size> nn2_input;
  // Matrix<float, batch_size, output_size> ground_truth_output;

  // TRAINING FROM BOTH NEURAL NETWORKS SEPARATELY
  // int iterations = 20000000;
  // for ( int i = 0; i < iterations/2; i++ ) {
    
  //     // NEURAL NETWORK 1
  //     float tempo = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 210 + 30;
  //     input = gen_time( tempo, offset, noise );
  //     ground_truth_output(0,0) = 60.0/tempo;
  //     nn1->apply(input);
  //     nn1->gradient_descent( input, ground_truth_output, true );

  //     // NEURAL NETWORK 2
  //     nn2_input( 0, 0 ) = 60.0/tempo;
  //     ground_truth_output(0,0) = tempo;
  //     nn2->apply(nn2_input);
  //     nn2->gradient_descent( nn2_input, ground_truth_output, true );

  //     // PRINT TRAINING PROGRESS
  //     if (i % 10000 == 0) {
  //       cout << (float(i)/iterations)*100 << " percent done..." << endl;
  //     }
  // }
  // nn1->printWeights("nn1_weights");
  // nn2->printWeights("nn2_weights");


  // INITIALIZE LARGE NEURAL NETWORK
  auto big_nn = make_unique<
      NeuralNetwork<float, big_nn_num_layers, batch_size, big_nn_input_size, output_size, 16, 1, 480, 480, output_size>>();
  big_nn->initialize(0.0000001);
  string weights_file = "/home/gbishko/nnfun/src/frontend/big_nn_1_weights.txt";
  big_nn->init_params(weights_file);



  /******************** TESTING ********************/
  Matrix<float, batch_size, big_nn_input_size> input;
  bool offset = true;
  bool noise = true;

  for (  int tempo = 30; tempo < 241; tempo++ ) {
    // Input is 16 time stamps, returns tempo
    input = gen_time( tempo, offset, noise);
    big_nn->apply(input);
    cout << tempo << " -> " << big_nn->get_output()(0,0) << endl;
  }
  cout << endl;

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



