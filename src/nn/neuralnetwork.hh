#pragma once
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Werror=effc++"

#include "network.hh"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// note that output_size need to be input twice,
// once as output_size, once in rest
template<class T,
         unsigned int num_of_layers,
         unsigned int batch_size,
         unsigned int input_size,
         unsigned int output_size,
         unsigned int... rest>
class NeuralNetwork
{

public:
  Network<T, batch_size, input_size, rest...>* nn {};

  // getters
  unsigned int get_num_of_layers() { return num_of_layers; }

  unsigned int get_input_size() { return input_size; }

  unsigned int get_output_size() { return output_size; }

  // default: initialize weights and biases randomly
  void initialize()
  {
    nn = new Network<T, batch_size, input_size, rest...>();
    nn->initializeWeightsRandomly();
  }

  void print() { nn->print(); }

  void apply( const Matrix<T, batch_size, input_size>& input ) { nn->apply( input ); }

  const Matrix<T, batch_size, output_size>& get_output() { return nn->output(); }

  float compute_pd_loss_wrt_output( const float target, const float actual ) { return -2 * ( target - actual ); }

  void gradient_descent( Matrix<T, batch_size, input_size>& input,
                         Matrix<T, batch_size, output_size>& ground_truth_output,
                         float learning_rate )
  {
    nn->apply( input );
    nn->computeDeltas();
    nn->evaluateGradients( input );

    float pd_loss_wrt_output = 0;
    // assuming batch_size = 1
    for ( int i = 0; i < (int)output_size; i++ ) {
      pd_loss_wrt_output += compute_pd_loss_wrt_output( ground_truth_output( 0, i ), nn->output()( 0, i ) );
    }

    // currently not dynamic learning rate
    for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
      for ( int j = 0; j < (int)nn->getNumParams( i ); j++ ) {
        nn->modifyParam(
          (unsigned int)i, j, learning_rate * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
      }
    }
  }
};
