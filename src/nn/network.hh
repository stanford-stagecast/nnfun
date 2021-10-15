#pragma once

#include "layer.hh"
#include <Eigen/Dense>
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
