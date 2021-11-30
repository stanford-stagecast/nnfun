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
  void apply( const Matrix<double, batch_size, i0>& input )
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

  double getEvaluatedGradient( const unsigned int layerNum, const unsigned int paramNum )
  {
    if ( layerNum > 0 ) {
      return next.getEvaluatedGradient( layerNum - 1, paramNum );
    }

    assert( layerNum == 0 );
    return layer0.getEvaluatedGradient( paramNum );
  }

  double calculateNumericalGradient( const Matrix<double, batch_size, i0>& input,
                                     const unsigned int layerNum,
                                     const unsigned int weightNum,
                                     const double epsilon = 1e-8 )
  {
    if ( layerNum > 0 ) {
      layer0.apply( input );
      return next.calculateNumericalGradient( layer0.output(), layerNum - 1, weightNum, epsilon );
    }

    assert( layerNum == 0 );
    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );
    Matrix<double, batch_size, output_size> fXPlusEpsilon = output();

    // f(X-epsilon)
    layer0.perturbWeight( weightNum, -2 * epsilon );
    apply( input );
    Matrix<double, batch_size, output_size> fXMinusEpsilon = output();

    // f(X) restore network to original state
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );

    const Matrix<double, batch_size, output_size>& derivative
      = ( fXPlusEpsilon - fXMinusEpsilon ) / ( 2 * epsilon );
    return derivative.sum() / batch_size;
  }

  unsigned int getLayerInputSize( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getLayerInputSize( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getInputSize();
  }
  unsigned int getLayerOutputSize( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getLayerOutputSize( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getOutputSize();
  }

  const Matrix<double, batch_size, i0> computeDeltas()
  {
    Matrix<double, batch_size, o0> nextLayerDeltas = next.computeDeltas();
    return layer0.computeDeltas( nextLayerDeltas );
  }

  void evaluateGradients( const Matrix<double, batch_size, i0>& input )
  {
    layer0.evaluateGradients( input );
    next.evaluateGradients( layer0.output() );
  }

  const Matrix<double, batch_size, output_size>& output() const { return next.output(); }
};

// BASE CASE
template<unsigned int batch_size, unsigned int i0, unsigned int o0>
class Network<batch_size, i0, o0>
{
public:
  Layer<batch_size, i0, o0> layer0 {};

  constexpr static unsigned int output_size = o0;

  void initializeWeightsRandomly() { layer0.initializeWeightsRandomly(); }

  void apply( const Matrix<double, batch_size, i0>& input ) { layer0.apply_without_activation( input ); }

  void print( const unsigned int layerNum = 0 ) const { layer0.print( layerNum ); }

  unsigned int getNumLayers() const { return 1; }

  unsigned int getNumParams( const unsigned int layerNum ) const
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getNumParams();
  }

  double getEvaluatedGradient( const unsigned int layerNum, const unsigned int paramNum )
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getEvaluatedGradient( paramNum );
  }

  double calculateNumericalGradient( const Matrix<double, batch_size, i0>& input,
                                     const unsigned int layerNum,
                                     const unsigned int weightNum,
                                     const double epsilon = 1e-8 )
  {
    assert( layerNum == 0 );
    (void)layerNum;
    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );
    Matrix<double, batch_size, output_size> fXPlusEpsilon = output();

    // f(X-epsilon)
    layer0.perturbWeight( weightNum, -2 * epsilon );
    apply( input );
    Matrix<double, batch_size, output_size> fXMinusEpsilon = output();

    // f(X) restore network to original state
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );

    const Matrix<double, batch_size, output_size>& derivative
      = ( fXPlusEpsilon - fXMinusEpsilon ) / ( 2 * epsilon );
    return derivative.sum() / batch_size;
  }

  unsigned int getLayerInputSize( const unsigned int layerNum ) const
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getInputSize();
  }

  unsigned int getLayerOutputSize( const unsigned int layerNum ) const
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getOutputSize();
  }

  const Matrix<double, batch_size, i0> computeDeltas()
  {
    Matrix<double, batch_size, o0> nextLayerDeltas = Matrix<double, batch_size, o0>::Ones() / batch_size;
    return layer0.computeDeltasLastLayer( nextLayerDeltas );
  }

  void evaluateGradients( const Matrix<double, batch_size, i0>& input ) { layer0.evaluateGradients( input ); }

  const Matrix<double, batch_size, o0>& output() const { return layer0.output(); }
};
