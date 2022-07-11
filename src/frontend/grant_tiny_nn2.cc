// Experimental 2nd TINY neural network to figure out how to manually change weights
#include "neuralnetwork.hh"
#include <Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <random>
#include "timer.hh"

using namespace std;
using namespace Eigen;

using std::cout; using std::cin;
using std::endl; using std::string;
using std::filesystem::current_path;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 3;

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 1, 1, 1>>();
  nn->initialize(); // default eta is 0.001

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;

  //input << 1;

  string printed_weights_file = "/home/gbishko/nnfun/src/frontend/output.txt";

  cout << "Current working directory: " << current_path() << endl;
  
  bool pre_trained = true;
  if (pre_trained) {
    nn->init_params(printed_weights_file);
  } else {
    /******************** TRAINING ********************/
    // NN should solve for 2X
    for ( int i = 0; i < 10000; i++ ) { 
      input(0,0) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) );
      ground_truth_output(0,0) = 2 * input(0,0);
      nn->apply(input);
      nn->gradient_descent( input, ground_truth_output, true );
      cout << "training " << i << endl;
    }
  }

  


  /******************** TESTING ********************/
  for ( float i = 0; i < 1.0; i += 0.05 ) { 
    input(0,0) = i;
    nn->apply( input );
    auto expected_output = 2 * input(0,0);
    auto nn_output = nn->get_output()(0,0);

    // PRINT RESULTS
    cout << input(0,0) << " || should be: " << expected_output << ", got -> " << nn_output << endl;
  }
  nn->print();
  //nn->printWeights();
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
