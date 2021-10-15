#pragma once

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

template<unsigned int batch_size_, unsigned int input_size_, unsigned int output_size_>
class Layer
{
private:
  Matrix<float, batch_size_, output_size_> output_ = {};
  Matrix<float, input_size_, output_size_> weights_ = Matrix<float, input_size_, output_size_>::Random();

public:
  Layer() {}
  void apply( const Matrix<float, batch_size_, input_size_>& input )
  {
    output_ = ( input * weights_ ).cwiseMax( 0 );
  }

  void apply_without_activation( const Matrix<float, batch_size_, input_size_>& input )
  {
    output_ = ( input * weights_ );
  }

  void print( const unsigned int layer_num ) const
  {
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

    cout << "Layer " << layer_num << endl;
    cout << "input size: " << input_size_ << " -> "
         << "output_size: " << output_size_ << endl
         << endl;

    cout << "weights:" << endl << weights().format( CleanFmt ) << endl << endl;
    cout << "output:" << endl << output().format( CleanFmt ) << endl << endl << endl;
  }

  const Matrix<float, input_size_, output_size_>& weights() const { return weights_; }
  const Matrix<float, batch_size_, output_size_>& output() const { return output_; }

  static constexpr unsigned int input_size() { return input_size_; }
  static constexpr unsigned int output_size() { return output_size_; }
  static constexpr unsigned int batch_size() { return batch_size_; }
};
