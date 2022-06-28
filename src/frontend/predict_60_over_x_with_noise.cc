#include "neuralnetwork.hh"

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

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, 480, 480, output_size>>();
  nn->initialize();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  // input = Matrix<float, batch_size, input_size>::Random();
  input << 1;
  ground_truth_output << 60.0 / 1;
  // nn->apply( input );
  // cout << nn->get_output()( 0, 0 ) << endl;
  // const uint64_t b_start = Timer::timestamp_ns();

  for ( int i = 0; i < 100000; i++ ) {
    float number = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 1.75 + 0.25;
    float noise = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * (0.1 * number) - (0.05 * number);
    input( 0, 0 ) = number + noise;
    /* if(i % 1000 == 0)
    {
      cout << input( 0, 0 ) << endl;
    } */
    ground_truth_output( 0, 0 ) = 60.0 / input( 0, 0 );
    nn->apply(input);
    // cout << i << " " << input(0,0) << " " << ground_truth_output(0,0) <<  " -> " << nn->get_output()(0,0) << endl;
    nn->gradient_descent( input, ground_truth_output, true );
  }

  // const uint64_t b_end = Timer::timestamp_ns();
  // cout << "timer: " << ( b_end - b_start ) << endl;

  // nn->print();
  //  testing
  for ( float i = 0.25; i < 2.0; i += 0.05 ) {
    float number = i;
    float noise = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * (0.1 * number) - (0.05 * number);
    input( 0, 0 ) = number + noise;
    nn->apply( input );
    cout << i << " " << (60.0/input(0,0)) << " -> " << nn->get_output()( 0, 0 ) << endl;
    // cout << i << " , " << (60.0/input(0,0)) << " , " << nn->get_output()( 0, 0 ) << endl;
    //cout << nn->get_output()( 0, 0 ) << endl;
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
    return EXIT_FAILURE;
  }
}