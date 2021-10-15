#include "timer.hh"
#include <Eigen/Dense>
#include <array>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

template<unsigned int n0, unsigned int... rest>
struct last_element
{
  constexpr static unsigned int value = last_element<rest...>::value;
};

template<unsigned int n>
struct last_element<n>
{
  constexpr static unsigned int value = n;
};

template<unsigned int batch_size_, unsigned int input_size_, unsigned int output_size_>
class Layer
{
private:
  Matrix<float, batch_size_, output_size_> output_ = {};
  Matrix<float, batch_size_, output_size_> unactivated_output_ = {};
  Matrix<float, input_size_, output_size_> weights_ = Matrix<float, input_size_, output_size_>::Random();

public:
  void apply( const Matrix<float, batch_size_, input_size_>& input )
  {
    unactivated_output_ = input * weights_;
    output_ = unactivated_output_.cwiseMax( 0 );
  }

  void print( int layer_num, bool is_final_layer = false )
  {
    const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

    cout << "Layer " << layer_num << endl;
    cout << "input size: " << input_size_ << " -> "
         << "output_size: " << output_size_ << endl
         << endl;

    cout << "weights:" << endl << weights_.format( CleanFmt ) << endl << endl;
    cout << "unactivated_output:" << endl << unactivated_output_.format( CleanFmt ) << endl << endl;
    if ( not is_final_layer ) {
      cout << "output:" << endl << output_.format( CleanFmt ) << endl << endl << endl;
    }
  }

  const Matrix<float, batch_size_, output_size_>& output() const { return output_; }
  const Matrix<float, batch_size_, output_size_>& unactivated_output() const { return unactivated_output_; }

  unsigned int input_size() const { return input_size_; }
  unsigned int output_size() const { return output_size_; }
  unsigned int batch_size() const { return batch_size_; }
};

template<unsigned int b, unsigned int i0, unsigned int o0, unsigned int... rest>
class Network
{
public:
  constexpr static unsigned int input_size = i0;
  constexpr static unsigned int output_size = last_element<o0, rest...>::value;
  constexpr static unsigned int batch_size = b;

  Layer<b, i0, o0> layer0 = {};
  Network<b, o0, rest...> next = {};

  void apply( const Matrix<float, b, i0>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  void print( int layer_num = 1 )
  {
    layer0.print( layer_num );
    next.print( layer_num + 1 );
  }

  const Matrix<float, b, output_size>& output() const { return next.output(); }
};

// BASE CASE
template<unsigned int b, unsigned int i0, unsigned int o0>
class Network<b, i0, o0>
{
public:
  constexpr static unsigned int input_size = i0;
  constexpr static unsigned int output_size = o0;
  constexpr static unsigned int batch_size = b;

  Layer<b, i0, o0> layer0 = {};
  void apply( const Matrix<float, b, i0>& input ) { layer0.apply( input ); }
  const Matrix<float, b, o0>& output() const { return layer0.unactivated_output(); }

  void print( int layer_num = 1 ) { layer0.print( layer_num, true ); }
};

int main()
{
  Network<2, 5, 3, 2, 1> nn;

  Matrix<float, 2, 5> inputs;
  inputs << 1, 2, 3, 4, 5, -1, -2, -3, -4, -5;

  nn.apply( inputs );
  auto& outputs = nn.output();

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

  cout << "input:" << endl << inputs.format( CleanFmt ) << endl << endl;
  nn.print();
  cout << "output:" << endl << outputs.format( CleanFmt ) << endl;

  return 0;
}
