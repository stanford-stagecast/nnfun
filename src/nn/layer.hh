#pragma once

#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

template<unsigned int batch_size_, unsigned int input_size_, unsigned int output_size_>
class Layer
{
private:
  Matrix<float, batch_size_, output_size_> output_ = {};
  Matrix<float, batch_size_, output_size_> unactivated_output_ = {};
  Matrix<float, input_size_, output_size_> weights_ = Matrix<float, input_size_, output_size_>::Random();
  bool is_final_layer_ = false;

public:
  Layer( bool is_final_layer ) { is_final_layer_ = is_final_layer; }
  Layer() {}
  void apply( const Matrix<float, batch_size_, input_size_>& input )
  {
    unactivated_output_ = input * weights_;
    if ( not is_final_layer_ ) {
      output_ = unactivated_output_.cwiseMax( 0 );
    } else {
      output_ = unactivated_output_;
    }
  }

  void print( int layer_num )
  {
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

    cout << "Layer " << layer_num << endl;
    cout << "input size: " << input_size_ << " -> "
         << "output_size: " << output_size_ << endl
         << endl;

    cout << "weights:" << endl << weights_.format( CleanFmt ) << endl << endl;
    cout << "unactivated_output:" << endl << unactivated_output_.format( CleanFmt ) << endl << endl;
    cout << "output:" << endl << output_.format( CleanFmt ) << endl << endl << endl;
  }

  const Matrix<float, batch_size_, output_size_>& output() const { return output_; }
  const Matrix<float, batch_size_, output_size_>& unactivated_output() const { return unactivated_output_; }

  unsigned int input_size() const { return input_size_; }
  unsigned int output_size() const { return output_size_; }
  unsigned int batch_size() const { return batch_size_; }
};
