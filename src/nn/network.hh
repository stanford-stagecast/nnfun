/**
 * Filename: network.hh
 * Last Update: June 2022
 */

#pragma once

#include "layer.hh"

using namespace std;
using namespace Eigen;

/*
 * Class Name: Network
 * Description: This is the recursive class template of neural networks. It
 *              creates layers specified by the user.
 *              It is able to initialize all weights and biases randomly.
 *              It is able to apply user input and get the corresponding output.
 *              It is able to compute delta and gradient.
 *              It has a print function which can beautifully print info.
 *
 * Inputs:
 *			1. T specifies the type of variables in the network (usually float)
 *			2. batch_size specifies the size of a batch (usually 1)
 *			3. i0 specifies the input size of the first layer
 *			4. o0 specifies the output size of the first layer | input size of
 * 			   the second layer
 *			5. rest... specifies the size of all of the later layers
 */
// RECURSIVE DEF
template<class T, unsigned int batch_size, unsigned int i0, unsigned int o0, unsigned int... rest>
class Network
{
public:
  /* two ctors */
  Layer<T, batch_size, i0, o0> layer0 {};
  Network<T, batch_size, o0, rest...> next {};

  constexpr static unsigned int output_size = decltype( next )::output_size;

  /*
   * FunctionName: initializeWeightsRandomly
   * Description: This function randomly recursively assigns values to all
   *              parameters in the neural network.
   */
  void initializeWeightsRandomly()
  {
    layer0.initializeWeightsRandomly();
    next.initializeWeightsRandomly();
  }

  void initializeWeights( const unsigned int layerNum, const Matrix<T, Dynamic, Dynamic>& weights )
  {
    if ( layerNum > 0 ) {
      next.initializeWeights( layerNum - 1, weights );
      return;
    }
    assert( layerNum == 0 );
    layer0.initializeWeights( weights );
  }

  void initializeBiases( const unsigned int layerNum, const Matrix<T, Dynamic, Dynamic>& biases )
  {
    if ( layerNum > 0 ) {
      next.initializeBiases( layerNum - 1, biases );
      return;
    }
    assert( layerNum == 0 );
    layer0.initializeBiases( biases );
  }

  /*
   * Function Name: aply
   * Description: This function recursively applys the input to the neuralnetwork.
   * Parameters:
   *			1. input is the input to the neural network
   */
  void apply( const Matrix<T, batch_size, i0>& input )
  {
    layer0.apply( input );
    next.apply( layer0.output() );
  }

  /*
   * Function Name: print
   * Description: This function prints the basic info of the neural network to
   *              stdout.
   * Parameters:
   *			1. layerNum specifies the position of the current layer to be
   *			   printed (starts with 0 and increments each time)
   */
  void print( const unsigned int layerNum = 0 ) const
  {
    layer0.print( layerNum );
    next.print( layerNum + 1 );
  }

  void printWeights ( const unsigned int layerNum = 0 ) const
  {
    layer0.printWeights (layerNum );
    next.printWeights( layerNum + 1 );
  }

  void printLayerOutput(const unsigned int layerNumber) const
  {
    if(layerNumber > 0)
    {
      next.printLayerOutput(layerNumber - 1);
    }
    else
    {
      layer0.printLayerOutput();
    }
  }

  /* getter of number of layers */
  unsigned int getNumLayers() const { return next.getNumLayers() + 1; }
  /* getter of number of parameters in a specific layer */
  unsigned int getNumParams( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getNumParams( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getNumParams();
  }

  /*
   * Function Name: modifyParamWholeLayer
   * Description: This function decrements all parameters by amount epsilon in
   *              the specified layer. It recursively finds the layer.
   * Parameters:
   *			1. layerNum specifies which layer to be decrementd
   *			2. epsilon is the constant to be decremented
   */
  void modifyParamWholeLayer( const unsigned int layerNum, T epsilon )
  {
    if ( layerNum > 0 ) {
      next.modifyParamWholeLayer( layerNum - 1, epsilon );
      return;
    }
    assert( layerNum == 0 );
    layer0.modifyParamWholeLayer( epsilon );
  }

  /*
   * Function Name: getEvaluatedGradient
   * Description: This function returns the previously computed gradient of the
   *              specified param in the specified layer.
   * Parameters:
   * 			1. layerNum specifies which layer
   *			2. paramNum specifies which parameter
   * Return Value: the gradient of type T (usually float)
   */
  T getEvaluatedGradient( const unsigned int layerNum, const unsigned int paramNum )
  {
    if ( layerNum > 0 ) {
      return next.getEvaluatedGradient( layerNum - 1, paramNum );
    }

    assert( layerNum == 0 );
    return layer0.getEvaluatedGradient( paramNum );
  }

  T calculateNumericalGradient( const Matrix<T, batch_size, i0>& input,
                                const unsigned int layerNum,
                                const unsigned int weightNum,
                                const T epsilon = 1e-8 )
  {
    if ( layerNum > 0 ) {
      layer0.apply( input );
      return next.calculateNumericalGradient( layer0.output(), layerNum - 1, weightNum, epsilon );
    }

    assert( layerNum == 0 );
    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );
    Matrix<T, batch_size, output_size> fXPlusEpsilon = output();

    // f(X-epsilon)
    layer0.perturbWeight( weightNum, -2 * epsilon );
    apply( input );
    Matrix<T, batch_size, output_size> fXMinusEpsilon = output();

    // f(X) restore network to original state
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );

    const Matrix<T, batch_size, output_size>& derivative = ( fXPlusEpsilon - fXMinusEpsilon ) / ( 2 * epsilon );
    return derivative.sum() / batch_size;
  }

  /* getter of input size of the specified layer */
  unsigned int getLayerInputSize( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getLayerInputSize( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getInputSize();
  }
  /* getter of output size of the specified layer */
  unsigned int getLayerOutputSize( const unsigned int layerNum ) const
  {
    if ( layerNum > 0 ) {
      return next.getLayerOutputSize( layerNum - 1 );
    }
    assert( layerNum == 0 );
    return layer0.getOutputSize();
  }

  const Matrix<T, batch_size, i0> computeDeltas()
  {
    Matrix<T, batch_size, o0> nextLayerDeltas = next.computeDeltas();
    return layer0.computeDeltas( nextLayerDeltas );
  }

  void evaluateGradients( const Matrix<T, batch_size, i0>& input )
  {
    layer0.evaluateGradients( input );
    next.evaluateGradients( layer0.output() );
  }

  const Matrix<T, batch_size, output_size>& output() const { return next.output(); }
};

/*
 * Class Name: Network
 * Description: This is the base case of the above recursive template.
 */
// BASE CASE
template<class T, unsigned int batch_size, unsigned int i0, unsigned int o0>
class Network<T, batch_size, i0, o0>
{
public:
  Layer<T, batch_size, i0, o0> layer0 {};

  constexpr static unsigned int output_size = o0;

  void initializeWeightsRandomly() { layer0.initializeWeightsRandomly(); }

  void initializeWeights( const unsigned int layerNum, const Matrix<T, i0, o0>& weights )
  {
    assert( layerNum == 0 );
    layer0.initializeWeights( weights );
  }

  void initializeBiases( const unsigned int layerNum, const Matrix<T, 1, o0>& biases )
  {
    assert( layerNum == 0 );
    layer0.initializeBiases( biases );
  }

  void apply( const Matrix<T, batch_size, i0>& input ) { layer0.apply_without_activation( input ); }

  void print( const unsigned int layerNum = 0 ) const { layer0.print( layerNum ); }

  void printWeights( const unsigned int layerNum = 0) const { layer0.printWeights( layerNum);}

  void printLayerOutput( const unsigned int layerNumber) const {int jonathan = layerNumber; jonathan++; layer0.printLayerOutput();}

  unsigned int getNumLayers() const { return 1; }

  unsigned int getNumParams( const unsigned int layerNum ) const
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getNumParams();
  }

  void modifyParamWholeLayer( const unsigned int layerNum, T epsilon )
  {
    assert( layerNum == 0 );
    layer0.modifyParamWholeLayer( epsilon );
  }

  T getEvaluatedGradient( const unsigned int layerNum, const unsigned int paramNum )
  {
    assert( layerNum == 0 );
    (void)layerNum;
    return layer0.getEvaluatedGradient( paramNum );
  }

  T calculateNumericalGradient( const Matrix<T, batch_size, i0>& input,
                                const unsigned int layerNum,
                                const unsigned int weightNum,
                                const T epsilon = 1e-8 )
  {
    assert( layerNum == 0 );
    (void)layerNum;
    // f(X+epsilon)
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );
    Matrix<T, batch_size, output_size> fXPlusEpsilon = output();

    // f(X-epsilon)
    layer0.perturbWeight( weightNum, -2 * epsilon );
    apply( input );
    Matrix<T, batch_size, output_size> fXMinusEpsilon = output();

    // f(X) restore network to original state
    layer0.perturbWeight( weightNum, epsilon );
    apply( input );

    const Matrix<T, batch_size, output_size>& derivative = ( fXPlusEpsilon - fXMinusEpsilon ) / ( 2 * epsilon );
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

  const Matrix<T, batch_size, i0> computeDeltas()
  {
    Matrix<T, batch_size, o0> nextLayerDeltas = Matrix<T, batch_size, o0>::Ones() / batch_size;
    return layer0.computeDeltasLastLayer( nextLayerDeltas );
  }

  void evaluateGradients( const Matrix<T, batch_size, i0>& input ) { layer0.evaluateGradients( input ); }

  const Matrix<T, batch_size, o0>& output() const { return layer0.output(); }
};
