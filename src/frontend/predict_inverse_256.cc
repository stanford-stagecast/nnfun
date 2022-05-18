#include "neuralnetwork.hh"

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 4;

void program_body()
{
  // num_layer = 2
  // batch_size = 1
  // input_size = 6
  // output_size = 1
  // 16 -> 16 -> 1
  // 2 1 16 1 16 1
  auto nn = make_unique<NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 30, 2560, 10, output_size>>();
  nn->initialize();
  nn->print();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  // input = Matrix<float, batch_size, input_size>::Random();
  input << 3;
  ground_truth_output << 1.0 / 3;
  // nn->apply( input );
  // cout << nn->get_output()( 0, 0 ) << endl;
  for ( int i = 0; i < 1000; i++ ) {
    input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 20 + 1;
    ground_truth_output( 0, 0 ) = 1.0 / input( 0, 0 );
    nn->gradient_descent( input, ground_truth_output, 0.1 );
  }

  nn->print();
  input( 0, 0 ) = 5;
  nn->apply( input );
  cout << nn->get_output()( 0, 0 ) << endl;
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