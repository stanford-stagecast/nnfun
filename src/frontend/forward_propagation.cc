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
  Matrix<float, output_size_, batch_size_> output_ = {};
  Matrix<float, output_size_, batch_size_> unactivated_output_ = {};
  Matrix<float, output_size_, input_size_> weights = Matrix<float, output_size_, input_size_>::Random();

public:
  void apply( const Matrix<float, input_size_, batch_size_>& input )
  {
    unactivated_output_ = weights * input;
    output_ = unactivated_output_.cwiseMax( 0 );
  }

  const Matrix<float, output_size_, batch_size_>& output() const { return output_; }
  const Matrix<float, output_size_, batch_size_>& unactivated_output() const { return unactivated_output_; }

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

  void apply( const Matrix<float, i0, b>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  const Matrix<float, output_size, b>& output() const { return next.output(); }
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
  void apply( const Matrix<float, i0, b>& input ) { layer0.apply( input ); }
  const Matrix<float, o0, b>& output() const { return layer0.unactivated_output(); }
};

int main()
{
  Network<2, 5, 3, 2, 1> nn;

  Matrix<float, 5, 2> inputs;
  inputs << 1, -1, 2, -2, 3, -3, 4, -4, 5, -5;

  nn.apply( inputs );
  auto& outputs = nn.output();

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

  cout << "input size: " << nn.input_size << endl;
  cout << nn.layer0.input_size() << " -> " << nn.layer0.output_size() << endl;
  cout << nn.next.layer0.input_size() << " -> " << nn.next.layer0.output_size() << endl;
  cout << nn.next.next.layer0.input_size() << " -> " << nn.next.next.layer0.output_size() << endl;
  cout << "output size: " << nn.output_size << endl;

  cout << outputs.format( CleanFmt ) << "\n";

  return 0;
}
