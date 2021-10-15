#pragma once

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

template<unsigned int batch_size, unsigned int input_size, unsigned int output_size>
class Layer
{
private:
  Matrix<float, batch_size, output_size> output_ {};
  Matrix<float, input_size, output_size> weights_ = Matrix<float, input_size, output_size>::Random();

public:
  Layer() {}
  void apply( const Matrix<float, batch_size, input_size>& input ) { output_ = ( input * weights_ ).cwiseMax( 0 ); }

  void apply_without_activation( const Matrix<float, batch_size, input_size>& input )
  {
    output_ = ( input * weights_ );
  }

  void print( const unsigned int layer_num ) const
  {
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

    cout << "Layer " << layer_num << endl;
    cout << "input size: " << input_size << " -> "
         << "output_size: " << output_size << endl
         << endl;

    cout << "weights:" << endl << weights().format( CleanFmt ) << endl << endl;
    cout << "output:" << endl << output().format( CleanFmt ) << endl << endl << endl;
  }

  const Matrix<float, input_size, output_size>& weights() const { return weights_; }
  const Matrix<float, batch_size, output_size>& output() const { return output_; }
};
