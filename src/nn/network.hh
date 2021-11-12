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
  void initializeWeightsRandomly()
  {
    layer0.initializeWeightsRandomly();
    next.initializeWeightsRandomly();
  }
  void apply( const Matrix<float, batch_size, i0>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  void print( const unsigned int layerNum = 0 ) const
  {
    layer0.print( layerNum );
    next.print( layerNum + 1 );
  }

  unsigned int getNumLayers() const { return next.getNumLayers() + 1; }

  unsigned int getNumParams( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getNumParams( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getNumParams();
  }

  float calculateNumericalGradient( const Matrix<float, batch_size, i0>& input,
                                    const unsigned int layerNum,
                                    const unsigned int weightNum,
                                    const float epsilon = 1e-8 )
  {
    if ( layerNum > 0 ) {
      layer0.apply( input );
      return next.calculateNumericalGradient( layer0.output(), layerNum - 1, weightNum, epsilon );
    }

    assert( layerNum == 0 );
    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );
    Matrix<float, batch_size, output_size> fXPlusEpsilon = output();

    // const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
    // cout << "input:" << endl << input.format( CleanFmt ) << endl;

    // cout << "fXPlusEpsilon" << endl;
    // print();

    // f(X)
    layer0.perturbWeight( weightNum, -epsilon );
    apply( input );
    Matrix<float, batch_size, output_size> fX = output();

    // cout << "input:" << endl << input.format( CleanFmt ) << endl;

    // cout << "fX" << endl;
    // print();
    const Matrix<float, batch_size, output_size>& derivative = ( fXPlusEpsilon - fX ) / epsilon;
    // cout << endl << endl << endl;
    return derivative.sum() / batch_size;
  }

  const Matrix<float, batch_size, output_size>& output() const { return next.output(); }
};

// BASE CASE
template<unsigned int batch_size, unsigned int i0, unsigned int o0>
class Network<batch_size, i0, o0>
{
public:
  Layer<batch_size, i0, o0> layer0 {};

  constexpr static unsigned int output_size = o0;

  void initializeWeightsRandomly() { layer0.initializeWeightsRandomly(); }

  void apply( const Matrix<float, batch_size, i0>& input ) { layer0.apply_without_activation( input ); }

  void print( const unsigned int layerNum = 0 ) const { layer0.print( layerNum ); }

  unsigned int getNumLayers() const { return 1; }

  unsigned int getNumParams( const unsigned int layerNum ) const
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getNumParams();
  }

  float calculateNumericalGradient( const Matrix<float, batch_size, i0>& input,
                                    const unsigned int layerNum,
                                    const unsigned int weightNum,
                                    const float epsilon = 1e-8 )
  {
    assert( layerNum == 0 );
    (void)layerNum;

    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );

    // const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
    // cout << "input:" << endl << input.format( CleanFmt ) << endl;

    Matrix<float, batch_size, o0> fXPlusEpsilon = output();
    // print();
    // cout << "fXPlusEpsilon" << fXPlusEpsilon.format( CleanFmt ) << endl;

    // f(X)
    layer0.perturbWeight( weightNum, -epsilon );
    apply( input );

    // cout << "input:" << endl << input.format( CleanFmt ) << endl;

    Matrix<float, batch_size, o0> fX = output();
    // print();
    // cout << "fX" << fX.format( CleanFmt ) << endl;
    Matrix<float, batch_size, o0> derivative = ( fXPlusEpsilon - fX ) / epsilon;

    return derivative.sum() / batch_size;
  }

  const Matrix<float, batch_size, o0>& output() const { return layer0.output(); }
};
