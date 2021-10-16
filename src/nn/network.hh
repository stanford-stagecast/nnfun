#pragma once

#include "layer.hh"

using namespace std;
using namespace Eigen;

template<unsigned int batch_size, unsigned int i0, unsigned int o0, unsigned int... rest>
class Network
{
public:
  Layer<batch_size, i0, o0> layer0 {};
  Network<batch_size, o0, rest...> next {};

  constexpr static unsigned int output_size = decltype( next )::output_size;

  void apply( const Matrix<float, batch_size, i0>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  void print( const unsigned int layer_num = 0 ) const
  {
    layer0.print( layer_num );
    next.print( layer_num + 1 );
  }

  const Matrix<float, batch_size, output_size>& output() const { return next.output(); }
};

// BASE CASE
template<unsigned int batch_size, unsigned int i0, unsigned int o0>
class Network<batch_size, i0, o0>
{
public:
  Layer<batch_size, i0, o0> layer0;

  constexpr static unsigned int output_size = o0;

  void apply( const Matrix<float, batch_size, i0>& input ) { layer0.apply_without_activation( input ); }

  void print( const unsigned int layer_num = 0 ) const { layer0.print( layer_num ); }

  const Matrix<float, batch_size, o0>& output() const { return layer0.output(); }
};
