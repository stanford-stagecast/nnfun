#include "layer.hh"
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

// template<unsigned int batch_size_, unsigned int input_size_, unsigned int output_size_>
// void Layer<batch_size_, input_size_, output_size_>::apply( const Matrix<float, batch_size_, input_size_>& input )
// {
//   unactivated_output_ = input * weights_;
//   output_ = unactivated_output_.cwiseMax( 0 );
// }

// template<unsigned int batch_size_, unsigned int input_size_, unsigned int output_size_>
// void Layer<batch_size_, input_size_, output_size_>::print( int layer_num, bool is_final_layer)
// {
//   const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

//   cout << "Layer " << layer_num << endl;
//   cout << "input size: " << input_size_ << " -> "
//        << "output_size: " << output_size_ << endl
//        << endl;

//   cout << "weights:" << endl << weights_.format( CleanFmt ) << endl << endl;
//   cout << "unactivated_output:" << endl << unactivated_output_.format( CleanFmt ) << endl << endl;
//   if ( not is_final_layer ) {
//     cout << "output:" << endl << output_.format( CleanFmt ) << endl << endl << endl;
//   }
// }