#include "timer.hh"
#include <Eigen/Dense>
#include <array>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Eigen;

struct NNParams
{
  unsigned int batch_size;
};

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

template<unsigned int input_size_, unsigned int output_size_, unsigned int batch_size_>
class Layer
{
private:
  Matrix<float, output_size_, batch_size_> output_;
  Matrix<float, output_size_, batch_size_> unactivated_output_;
  Matrix<float, output_size_, input_size_> weights;

public:
  void apply( const Matrix<float, input_size_, batch_size_>& input )
  {
    unactivated_output_ = weights * input;
    output_ = unactivated_output_;
  }
  const Matrix<float, output_size_, batch_size_>& output() const { return output_; }
  const Matrix<float, output_size_, batch_size_>& unactivated_output() const { return unactivated_output_; }

  unsigned int input_size() const { return input_size_; }
  unsigned int output_size() const { return output_size_; }
  unsigned int batch_size() const { return batch_size_; }
};

template<struct NNParams nn_params, unsigned int i0, unsigned int o0, unsigned int... rest>
class Network
{
public:
  constexpr static unsigned int input_size = i0;
  constexpr static unsigned int output_size = last_element<o0, rest...>::value;
  constexpr static unsigned int batch_size = nn_params.batch_size;

  Layer<i0, o0, nn_params.batch_size> layer0;
  Network<nn_params, o0, rest...> next;

  void apply( const Matrix<float, i0, nn_params.batch_size>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  const Matrix<float, output_size, nn_params.batch_size>& output() const { return next.output(); }
};

// BASE CASE
template<struct NNParams nn_params, unsigned int i0, unsigned int o0>
class Network<nn_params, i0, o0>
{
public:
  constexpr static unsigned int input_size = i0;
  constexpr static unsigned int output_size = o0;
  constexpr static unsigned int batch_size = nn_params.batch_size;

  Layer<i0, o0> layer0;
  void apply( const Matrix<float, i0, nn_params.batch_size>& input ) { layer0.apply( input ); }
  const Matrix<float, o0, nn_params.batch_size>& output() const { return layer0.unactivated_output(); }
};

int main()
{
  struct NNParams nn_params
  {
    2
  };
  // Network<NNParams{2}, 5, 3, 2, 1> nn;

  // cout << "input size: " << nn.input_size << endl;
  // cout << nn.layer0.input_size() << " -> " << nn.layer0.output_size() << endl;
  // cout << nn.next.layer0.input_size() << " -> " << nn.next.layer0.output_size() << endl;
  // cout << nn.next.next.layer0.input_size() << " -> " << nn.next.next.layer0.output_size() << endl;
  // cout << "output size: " << nn.output_size << endl;

  // Matrix<float, 5, 2> inputs {
  //   { 1, -1 }, { 2, -2 }, { 3, -3 }, { 4, -4 }, { 5, -5 },
  // };

  // nn.apply( input );
  // auto& output = nn.output();
  // (void)output;
  return 0;
}
