/**
 * Fila name: neuralnetwork.hh
 * Last Update: June 2022
 */
#pragma once

#include "network.hh"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
 * Class Name: NeuralNetwork
 * Description: This class defines the behavior of the whole neural network.
 *              It provides a randomly initialization function, forward
 *              propagation, and backward propagation.
 *              It provides a print function to visualize the whole neural network.
 */

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

public:
  Network<T, batch_size, input_size, rest...>* nn {};

  // getters
  unsigned int get_num_of_layers() { return num_of_layers; }

  unsigned int get_input_size() { return input_size; }

  unsigned int get_output_size() { return output_size; }

  float get_current_learning_rate() { return learning_rate; }

  const Matrix<T, batch_size, output_size>& get_output() { return nn->output(); }

  /*
   * Function Name: initialize
   * Description: This function initializes an object of Network class based on
   *              user inputs. Then it randomly assigns values to all parameters
   *              in the neural network.
   * Parameters:
   *			1. eta is the user-initialized learning rate, default 
   *			   to be 0.001
   */
  void initialize( float eta = 0.001 )
  {
    nn = new Network<T, batch_size, input_size, rest...>();
    nn->initializeWeightsRandomly();
    learning_rate = eta;
  }

  /*
   * Function Name: print
   * Description: This function prints the basic info of the whole neural network.
   */
  void print() { nn->print(); }

  /*
   * Function Name: apply
   * Description: This function applys the user input to the neural network.
   * Parameters:
   *			1. input is the input to the neural network
   */
  void apply( const Matrix<T, batch_size, input_size>& input ) { nn->apply( input ); }

  /*
   * Function Name: gradient_descent
   * Description: This function performs gradient descent based on the input
   *              and the groundtruth output. User can truth whether to conduct
   *              dynamic learning rate or not.
   * Parameters:
   *			1. input is the input used for gradient descent
   *			2. ground_truth_output is the actual value used to compare with
   *			   the predicted output in the loss function
   *			3. dynamic is a boolean value stating whether to do dynamic
   *			   learning rate or not
   */
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

      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        nn->modifyParamWholeLayer( i, learning_rate * pd_loss_wrt_output );
      }
    } else {
      // original
      float current_loss = 0;
      float pd_loss_wrt_output = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        pd_loss_wrt_output += compute_pd_loss_wrt_output( ground_truth_output( 0, i ), nn->output()( 0, i ) );
        current_loss += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      // 4/3 learning rate
      float lr_4_3 = 4.0 / 3 * learning_rate;
      // modifying weights and biases
      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        nn->modifyParamWholeLayer( i, lr_4_3 * pd_loss_wrt_output );
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
        nn->modifyParamWholeLayer( i, ( -lr_4_3 + lr_2_3 ) * pd_loss_wrt_output );
      }
      // compute loss
      nn->apply( input );
      float loss_2_3 = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        loss_2_3 += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      float min_loss = min( min( current_loss, loss_4_3 ), loss_2_3 );
      if ( min_loss == current_loss ) {
        //  update learning rate
        learning_rate = lr_2_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          nn->modifyParamWholeLayer( i, -lr_2_3 * pd_loss_wrt_output );
        }
      } else if ( min_loss == loss_4_3 ) {
        //  update learning rate
        learning_rate = lr_4_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          nn->modifyParamWholeLayer( i, ( -lr_2_3 + lr_4_3 ) * pd_loss_wrt_output );
        }
      } else {
        //  no need
      }
    }
  }
};
