#include "neuralnetwork.hh"

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 16;
constexpr size_t output_size = 1;

/* compute input */
Matrix<float, batch_size, input_size> gen_time( float tempo, float offset )
{
  Matrix<float, batch_size, input_size> ret_mat;
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = 60 / tempo * i + offset;
  }
  return ret_mat;
}


void program_body()
{
  /* construct neural network on heap */
  auto nn = make_unique<NeuralNetwork<float, 4, batch_size, input_size, output_size, 30, 2560, 10, output_size>>();
  nn->initialize();

  float offset = 0;

  for(int i = 0; i < 100000; i++) {
    float tempo = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) * 40 + 40;
    Matrix<float, batch_size, input_size> input = gen_time( tempo, offset );
    Matrix<float, batch_size, output_size> ground_truth_output;
    ground_truth_output(0,0) = tempo;
    nn->gradient_descent(input, ground_truth_output, true);
  }
  
    

  for ( int i = 40; i < 80; i++ ) {
    Matrix<float, batch_size, input_size> input = gen_time( i, 0 );
    nn->apply( input );
    //cout << "input: " << i << " output: " << nn->get_output()( 0, 0 ) << endl;
    cout << nn->get_output()(0,0) << endl;
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
