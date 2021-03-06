#include "neuralnetwork.hh"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>

#include "timer.hh"

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 4;

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 16, 256, 512, output_size>>();
  nn->initialize();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  input << 1;
  ground_truth_output << 4;

  // const uint64_t b_start = Timer::timestamp_ns();

  for ( int i = 0; i < 50000; i++ ) {
    input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 100 + 1;
    ground_truth_output( 0, 0 ) = 3 * input( 0, 0 ) + 1;
    nn->apply(input);
    nn->gradient_descent( input, ground_truth_output, true );
  }

  // const uint64_t b_end = Timer::timestamp_ns();
  // cout << "timer: " << ( b_end - b_start ) << endl;

  //  testing
  for ( float i = 1; i < 102; i ++ ) {
    input( 0, 0 ) = i;
    nn->apply( input );
    cout << (3 * input(0,0) + 1) << " -> " << nn->get_output()( 0, 0 ) << endl;
  }
  ofstream ofs{"../src/frontend/output_weights.txt"};
  auto cout_buff = cout.rdbuf(); 
  cout.rdbuf(ofs.rdbuf()); 
  nn->printWeights();
  cout.rdbuf(cout_buff);
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