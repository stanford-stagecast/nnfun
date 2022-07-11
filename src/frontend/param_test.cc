// Test reading file to initialize weights/biases
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
  auto nn_pre_trained = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 16, 1>>();
  auto nn_not_pre_trained = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 16, 1>>();
  nn_pre_trained->initialize();
  nn_not_pre_trained->initialize();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;


  /******************** TRAIN nn_not_pre_trained ********************/
  // NN should solve for 2X + 1
  for ( int i = 0; i < 100000; i++ ) { 
    input(0,0) = (static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) )) * 100;
    ground_truth_output(0,0) = (2 * input(0,0)) + 1;
    nn_not_pre_trained->apply(input);
    nn_not_pre_trained->gradient_descent( input, ground_truth_output, true );
  }
  nn_not_pre_trained->printWeights("trained_weights.txt");


 /* Load weights and biases of nn_not_pre_trained to nn_pre_trained */
  string printed_weights_file = "/home/gbishko/nnfun/src/frontend/trained_weights.txt";
  nn_pre_trained->init_params(printed_weights_file);
  


  /****** TEST/COMPARE THAT THE NNs BEHAVE THE SAME ******/
  for ( float i = 0; i < 100; i += 0.5 ) { 
    input(0,0) = i;
    nn_not_pre_trained->apply( input );
    nn_pre_trained->apply( input );

    auto expected_output = 2 * input(0,0) + 1;
    auto nn_pre_trained_output = nn_pre_trained->get_output()(0,0);
    auto nn_not_pre_trained_output = nn_not_pre_trained->get_output()(0,0);

    if (nn_pre_trained_output != nn_not_pre_trained_output) {
      cout << "ERROR " << input(0,0) << " || should be: " << expected_output << ", got -> " << nn_pre_trained_output << " and " << nn_not_pre_trained_output << endl;
      throw runtime_error("NEURAL NETWORKS BEHAVED DIFFERENTLY");
    }

    // PRINT RESULTS
    //cout << input(0,0) << " || should be: " << expected_output << ", got -> " << nn_pre_trained_output << " and " << nn_not_pre_trained_output << endl;
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
