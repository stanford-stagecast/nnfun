#pragma once

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

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

public:
  Layer() {}

  void initializeWeightsRandomly()
  {
    weights_ = Matrix<T, input_size, output_size>::Random();
    biases_ = Matrix<T, 1, output_size>::Random();
  }

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

  unsigned int getNumParams() const { return ( input_size + 1 ) * output_size; }
  unsigned int getInputSize() const { return input_size; }
  unsigned int getOutputSize() const { return output_size; }

  void modifyParam( const unsigned int paramNum, T newVal )
  {
    // cout << paramNum << " " << newVal << endl;
    const unsigned int i = paramNum / output_size;
    const unsigned int j = paramNum % output_size;
    if ( i < input_size ) {
      weights_( i, j ) -= newVal;
    } else {
      biases_( 0, j ) -= newVal;
    }
  }

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
