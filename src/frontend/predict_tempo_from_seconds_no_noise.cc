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
constexpr size_t num_layers = 4;
constexpr size_t layer_size = 1000;

Matrix<float, batch_size, input_size> gen_time( float tempo, float offset )
{
  Matrix<float, batch_size, input_size> ret_mat; //empty matrix [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = (60.0/tempo) * i + offset; //[0, 0.25, 0.5, 0.75, 1, 1.25, ...] (backward in time)
  }
  return ret_mat;
}

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size, layer_size, layer_size, output_size>>();
  nn->initialize();

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  // input = Matrix<float, batch_size, input_size>::Random();
  //input << 1;
  //ground_truth_output << 60.0 / 1;
  // nn->apply( input );
  // cout << nn->get_output()( 0, 0 ) << endl;
  // const uint64_t b_start = Timer::timestamp_ns();

  float offset = 0;
  for ( int i = 0; i < 100000; i++ ) {
    float tempo = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 260 + 10;
    input = gen_time( tempo, offset );
    // input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 1.75 + 0.25;
    // ground_truth_output( 0, 0 ) = 60.0 / input( 0, 0 );
    if(i % 1000 == 0)
    {
      cout << tempo << " apple pies loading..." << endl;
    }
    if (i == 99999)
    {
      cout << tempo << endl;
    }
    ground_truth_output(0,0) = tempo;
    nn->apply(input);
    nn->gradient_descent( input, ground_truth_output, true );

  }

  // const uint64_t b_end = Timer::timestamp_ns();
  // cout << "timer: " << ( b_end - b_start ) << endl;

  // nn->print();
  //  testing
  for (  int tempo = 30; tempo < 241; tempo++ ) {
    input = gen_time( tempo, offset );
    // input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 1.75 + 0.25;
    // ground_truth_output( 0, 0 ) = 60.0 / input( 0, 0 );
    nn->apply(input);
    cout << tempo << " -> " << nn->get_output()( 0, 0 ) << endl;
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