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

private:
  float learning_rate = 0.001;

  float compute_pd_loss_wrt_output( const float target, const float actual ) { return -2 * ( target - actual ); }

  float loss_function( const float target, const float actual )
  {
    return ( target - actual ) * ( target - actual );
  }

  // vector<num_>

public:
  Network<T, batch_size, input_size, rest...>* nn {};

  // getters
  unsigned int get_num_of_layers() { return num_of_layers; }

  unsigned int get_input_size() { return input_size; }

  unsigned int get_output_size() { return output_size; }

  unsigned int get_current_learning_rate() { return learning_rate; }

  // default: initialize weights and biases randomly
  void initialize()
  {
    nn = new Network<T, batch_size, input_size, rest...>();
    nn->initializeWeightsRandomly();
  }

  void print() { nn->print(); }

  void apply( const Matrix<T, batch_size, input_size>& input ) { nn->apply( input ); }

  const Matrix<T, batch_size, output_size>& get_output() { return nn->output(); }

  void gradient_descent( Matrix<T, batch_size, input_size>& input,
                         Matrix<T, batch_size, output_size>& ground_truth_output,
                         bool dynamic )
  {
    nn->apply( input );
    nn->computeDeltas();
    nn->evaluateGradients( input );

    if ( !dynamic ) {
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
    } else {
      // original
      float current_loss = 0;
      float pd_loss_wrt_output = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        pd_loss_wrt_output += compute_pd_loss_wrt_output( ground_truth_output( 0, i ), nn->output()( 0, i ) );
        current_loss += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }
      // cout << endl << pd_loss_wrt_output << endl;

      // 4/3 learning rate
      float lr_4_3 = 4.0 / 3 * learning_rate;
      // modifying weights and biases
      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        for ( int j = 0; j < (int)nn->getNumParams( i ); j++ ) {
          nn->modifyParam( (unsigned int)i, j, lr_4_3 * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
        }
      }
      // compute loss
      nn->apply( input );
      float loss_4_3 = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        loss_4_3 += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      // 2/3 learning rate
      float lr_2_3 = 2.0 / 3 * learning_rate;
      // modifying weights and biases
      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        for ( int j = 0; j < (int)nn->getNumParams( i ); j++ ) {
          nn->modifyParam(
            (unsigned int)i, j, ( -lr_4_3 + lr_2_3 ) * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
        }
      }
      // compute loss
      nn->apply( input );
      float loss_2_3 = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        loss_2_3 += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      float min_loss = min( min( current_loss, loss_4_3 ), loss_2_3 );
      if ( min_loss == current_loss ) {
        // cout << "current loss: " << current_loss << "       4/3: " << loss_4_3 << " 2/3: " << loss_2_3 << endl;
        //  update learning rate
        learning_rate = lr_2_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          for ( int j = 0; j < (int)nn->getNumParams( i ); j++ ) {
            nn->modifyParam( (unsigned int)i, j, -lr_2_3 * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
          }
        }
      } else if ( min_loss == loss_4_3 ) {
        // cout << "loss 4/3 " << loss_4_3 << "         current loss: " << current_loss << " 2/3: " << loss_2_3 <<
        // endl;
        //  update learning rate
        learning_rate = lr_4_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          for ( int j = 0; j < (int)nn->getNumParams( i ); j++ ) {
            nn->modifyParam(
              (unsigned int)i, j, ( -lr_2_3 + lr_4_3 ) * pd_loss_wrt_output * nn->getEvaluatedGradient( i, j ) );
          }
        }
      } else {
        // cout << "loss 2/3 " << loss_2_3 << "          current loss: " << current_loss << " 4/3: " << loss_4_3 <<
        // endl;
        //  no need
      }
    }
  }
};
