/**
 * File name: layer.hh
 * Last Update: June 2022
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

/*
 * Class Name: Layer
 * Description: This class defines the behavior of the layer (atom component of
 *              neural networks).
 *              It stores the info of input size, output size, the weights and
 *              biases.
 *              It is able to apply user input and get the corresponding output.
 *              It is able to compute delta and gradient.
 *              It is able to perform back propagation (modifyParamWholeLyaer
 *              and perturbWeight) indicated by the user.
 *              It has a print function to visualize.
 *
 *              Currently the version only supports layer with number of weight
 *              variables being input_size * output_size, and number of biase
 *              variables being output_size.
 *              Example: If there are x inputs to the layer, and the layer has
 *                       y outputs, then there will be x*y weights, and y biases.
 *
 * Inputs:
 *			1. T specifies the type of variables in the layer (usually float)
 *			2. batch_size specifies the size of a batch (usually 1)
 *			3. input_size specifies the size of input
 *			4. output_size specifies the size of output
 */
template<class T, unsigned int batch_size, unsigned int input_size, unsigned int output_size>
class Layer
{
private:
  // matrix to store outputs of neurons after activation sigma(W*X + B)
  Matrix<T, batch_size, output_size> output_ {};
  // matrix to store outputs of neurons before activation W*X + B
  Matrix<T, batch_size, output_size> unactivated_output_ {};
  // matrix to store weights of the connections  (W)*X + (B)
  Matrix<T, input_size, output_size> weights_ {};
  // matrix to store biases of the layer W*X + (B)
  Matrix<T, 1, output_size> biases_ {};

  // matrix to store errors at intermediate nodes for given input i.e. target activation - current activation
  Matrix<T, batch_size, output_size> deltas_ {};
  // matrix to store gradients w.r.t. weights
  Matrix<T, input_size, output_size> grad_weights_ {};
  // matrix to store gradients w.r.t. biases
  Matrix<T, 1, output_size> grad_biases_ {};

  // unsigned int numParam = (input_size + 1) * output_size;

public:
  Layer() {}

  /*
   * Function Name: initializeWeightsRandomly
   * Description: This function randomly assigns values in Matrix weights_ and
   *			  Matrix biases_.
   */
  void initializeWeightsRandomly()
  {
    weights_ = Matrix<T, input_size, output_size>::Random();
    biases_ = Matrix<T, 1, output_size>::Random();
  }

  /*
   * Function Name: apply
   * Description: This function applys the input to the neuralnetwork.
   *              The Matrix output_ will be updated.
   * Parameters:
   * 			1. input is the input to the layer
   */
  void apply( const Matrix<T, batch_size, input_size>& input )
  {
    unactivated_output_ = ( input * weights_ ).rowwise() + biases_;
    output_ = unactivated_output_.cwiseMax( 0 );
  }

  void apply_without_activation( const Matrix<T, batch_size, input_size>& input )
  {
    unactivated_output_ = ( input * weights_ ).rowwise() + biases_;
    output_ = unactivated_output_;
  }

  /*
   * Function Name: print
   * Description: This function prints the basic info of the layer to stdout.
   * Parameters:
   *			1. layer_num specifies the position of current layer in the
   *			   whole neural network (useful information in multi-layer nn)
   */
  void print( const unsigned int layer_num ) const
  {
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

    cout << "Layer " << layer_num << endl;
    cout << "input_size: " << input_size << " -> "
         << "output_size: " << output_size << endl
         << endl;

    cout << "weights:" << endl << weights_.format( CleanFmt ) << endl << endl;
    cout << "biases:" << endl << biases_.format( CleanFmt ) << endl << endl;
    cout << "unactivated_output:" << endl << unactivated_output_.format( CleanFmt ) << endl << endl;
    cout << "output:" << endl << output_.format( CleanFmt ) << endl << endl;

    cout << "deltas:" << endl << deltas_.format( CleanFmt ) << endl << endl;
    cout << "grad_weights:" << endl << grad_weights_.format( CleanFmt ) << endl << endl;
    cout << "grad_biases:" << endl << grad_biases_.format( CleanFmt ) << endl << endl << endl;
  }

  void printWeights( const unsigned int layer_num ) const
  {
    cout << layer_num << endl;
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
    cout << "weights:" << endl << weights_.format( CleanFmt ) << endl << endl;

  }

  /*
   * Function Name: perturbWeight
   * Description: This function increments the parameter (either weight in Matrix
   *              weights_ or biase in Matrix biases_) by amount epsilon.
   * Parameters:
   * 			1. weight_num specifies which parameter to be changed
   *			2. epsilon is the amount to be incremented
   */
  void perturbWeight( const unsigned int weight_num, const T epsilon )
  {
    const unsigned int i = weight_num / output_size;
    const unsigned int j = weight_num % output_size;
    if ( i < input_size ) {
      weights_( i, j ) += epsilon;
    } else {
      biases_( 0, j ) += epsilon;
    }
  }

  /* getter of number of parameter*/
  unsigned int getNumParams() const { return ( input_size + 1 ) * output_size; }
  /* getter of input size */
  unsigned int getInputSize() const { return input_size; }
  /* getter of output size */
  unsigned int getOutputSize() const { return output_size; }

  /*
   * Function Name: modifyParamWholeLayer
   * Description: This function decrements all parameters including weights in
   *		      Matrix weights_ and biases in Matrix biases_ by amount
   *              (epsilon * the gradient calculated at the corresponding location).
   * Parameters:
   *			1. epsilon is the constant to be multiplied to the amount to be
   *			   decremented
   */
  void modifyParamWholeLayer( T epsilon )
  {
    weights_ -= grad_weights_ * epsilon;
    biases_ -= grad_biases_ * epsilon;
  }

  /*
   * Function Name: getEvaluatedGradient
   * Description: This function returns the previously computed gradient of
   *              the specified param.
   * Parameters:
   *			1. paramNum specifies which parameter to return the gradient
   * Return Value: the gradient of type T (usually float)
   */
  T getEvaluatedGradient( const unsigned int paramNum )
  {
    const unsigned int i = paramNum / output_size;
    const unsigned int j = paramNum % output_size;
    if ( i < input_size ) {
      return grad_weights_( i, j );
    } else {
      return grad_biases_( 0, j );
    }
  }

  const Matrix<T, batch_size, input_size> computeDeltas( Matrix<T, batch_size, output_size> nextLayerDeltas )
  {
    // activated nodes is the matrix that stores 0/1 corresponding to whether the output node was activated
    Matrix<T, batch_size, output_size> activated_nodes
      = ( unactivated_output_.array() > 0 ).template cast<T>().matrix();
    deltas_ = nextLayerDeltas.cwiseProduct( activated_nodes );
    return deltas_ * weights_.transpose();
  }

  const Matrix<T, batch_size, input_size> computeDeltasLastLayer(
    Matrix<T, batch_size, output_size> nextLayerDeltas )
  {
    deltas_ = nextLayerDeltas;
    return deltas_ * weights_.transpose();
  }

  void evaluateGradients( const Matrix<T, batch_size, input_size>& input )
  {
    grad_weights_ = Matrix<T, input_size, output_size>::Zero();
    // grad_biases_ = Matrix<T, 1, output_size>::Zero();
    for ( unsigned int b = 0; b < batch_size; b++ ) {
      // for ( unsigned int j = 0; j < output_size; j++ ) {
      //   // for ( unsigned int i = 0; i < input_size; i++ ) {
      //   //   grad_weights_( i, j ) += input( b, i ) * deltas_( b, j );
      //   // }
      //   grad_weights_.col( j ) += input.row( b ) * deltas_( b, j );
      // }
      grad_weights_.noalias() += input.row( b ).transpose() * deltas_.row( b );
      // grad_biases_.noalias() += deltas_.row( b );
      // noalias is an eigen optimisation - otherwise becomes slower than for loops
    }
    grad_biases_ = deltas_.colwise().sum();
  }

  const Matrix<T, input_size, output_size>& weights() const { return weights_; }
  const Matrix<T, batch_size, output_size>& output() const { return output_; }
  const Matrix<T, 1, output_size>& biases() const { return biases_; }

  // accessors for mutable access to weights and biases
  Matrix<T, input_size, output_size>& weights() { return weights_; }
  Matrix<T, 1, output_size>& biases() { return biases_; }
};
