/**
 * Fila name: neuralnetwork.hh
 * Last Update: June 2022
 */
#pragma once

#include "network.hh"

#include <Eigen/Dense>
#include <regex>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace Eigen;

/*
 * Class Name: NeuralNetwork
 * Description: This class defines the behavior of the whole neural network.
 *              It provides a randomly initialization function, forward
 *              propagation, and backward propagation.
 *              It provides a print function to visualize the whole neural network.
 */

// note that output_size need to be input twice,
// once as output_size, once in rest
template<class T,
         unsigned int num_of_layers,
         unsigned int batch_size,
         unsigned int input_size,
         unsigned int output_size,
         unsigned int... rest>
class NeuralNetwork
{

private:
  float learning_rate = 0.001;

  float compute_pd_loss_wrt_output( const float target, const float actual ) { return -2 * ( target - actual ); }

  float loss_function( const float target, const float actual )
  {
    return ( target - actual ) * ( target - actual );
  }

  bool isNumber( const string& str )
  {
    for ( char c : str ) {
      if ( isdigit( c ) == 0 )
        return false;
    }
    return true;
  }

public:
  Network<T, batch_size, input_size, rest...>* nn {};

  // getters
  unsigned int get_num_of_layers() { return num_of_layers; }

  unsigned int get_input_size() { return input_size; }

  unsigned int get_output_size() { return output_size; }

  float get_current_learning_rate() { return learning_rate; }

  const Matrix<T, batch_size, output_size>& get_output() { return nn->output(); }

  /*
   * Function Name: initialize
   * Description: This function initializes an object of Network class based on
   *              user inputs. Then it randomly assigns values to all parameters
   *              in the neural network.
   * Parameters:
   *			1. eta is the user-initialized learning rate, default
   *			   to be 0.001
   */
  void initialize( float eta = 0.001 )
  {
    nn = new Network<T, batch_size, input_size, rest...>();
    nn->initializeWeightsRandomly();
    learning_rate = eta;
  }

  int init_params( string& filename )
  {
    ifstream file( filename );
    string line;
    int curr_layer = -1;
    bool is_weight = true;
    int cnt = 0;
    unsigned int curr_input_size = 0;
    unsigned int curr_output_size = 0;
    Matrix<T, Dynamic, Dynamic> weights_;
    Matrix<T, Dynamic, Dynamic> biases_;
    regex regex( ", " );
    while ( getline( file, line ) ) {
      // empty line -> update params
      if ( line.empty() )
        continue;
      // layer num -> get input size and output size
      else if ( isNumber( line ) ) {
        if ( curr_layer != -1 ) {
          nn->initializeWeights( curr_layer, weights_ );
          nn->initializeBiases( curr_layer, biases_ );
        }
        curr_layer = stoi( line );
        curr_input_size = nn->getLayerInputSize( curr_layer );
        curr_output_size = nn->getLayerOutputSize( curr_layer );
      }
      // contains "weights" -> next line(s) will be param for weight
      else if ( strstr( line.c_str(), "weights" ) ) {
        is_weight = true;
        cout << "here" << curr_input_size << " " << curr_output_size << endl;
        weights_.resize( (int)curr_input_size, (int)curr_output_size );
        cnt = 0;
      }
      // contains "biases" -> next line will be param for biase
      else if ( strstr( line.c_str(), "biases" ) ) {
        is_weight = false;
        cout << "this" << curr_output_size << endl;
        cout << biases_.rows() << " " << biases_.cols() << endl;
        biases_.resize( 1, (int)curr_output_size );
    } else {
      vector<string> params( sregex_token_iterator( line.begin(), line.end(), regex, -1 ),
                             sregex_token_iterator() );
      // get rid of square brackets
      auto s = &( params[0] );
      ( *s ).erase( remove( ( *s ).begin(), ( *s ).end(), '[' ), ( *s ).end() );
      s = &( params[params.size() - 1] );
      s->erase( remove( s->begin(), s->end(), ']' ), s->end() );

      for ( int i = 0; i < (int)params.size(); i++ ) {
        float p = stof( params[i] );
        cout << p << endl;
        if ( is_weight ) {
          weights_( cnt, i ) = p;
        } else {
          biases_( 0, i ) = p;
        }
      }
      cnt++;
    }
  }
  file.close();
  return 0;
}

  /*
   * Function Name: print
   * Description: This function prints the basic info of the whole neural network.
   */
  void print() { nn->print(); }

  void printWeights() { 
    ofstream ofs{"../src/frontend/output_weights.txt"};
    auto cout_buff = cout.rdbuf();
    cout.rdbuf(ofs.rdbuf());
    nn->printWeights();
    cout.rdbuf(cout_buff);
    }

  /*
   * Function Name: apply
   * Description: This function applys the user input to the neural network.
   * Parameters:
   *			1. input is the input to the neural network
   */
  void apply( const Matrix<T, batch_size, input_size>& input ) { nn->apply( input ); }

  /*
   * Function Name: gradient_descent
   * Description: This function performs gradient descent based on the input
   *              and the groundtruth output. User can truth whether to conduct
   *              dynamic learning rate or not.
   * Parameters:
   *			1. input is the input used for gradient descent
   *			2. ground_truth_output is the actual value used to compare with
   *			   the predicted output in the loss function
   *			3. dynamic is a boolean value stating whether to do dynamic
   *			   learning rate or not
   */
  void gradient_descent( Matrix<T, batch_size, input_size>& input,
                         Matrix<T, batch_size, output_size>& ground_truth_output,
                         bool dynamic )
  {
    nn->apply( input );
    nn->computeDeltas();
    nn->evaluateGradients( input );

    if ( !dynamic ) {
      float pd_loss_wrt_output = 0;
      // assuming batch_size = 1
      for ( int i = 0; i < (int)output_size; i++ ) {
        pd_loss_wrt_output += compute_pd_loss_wrt_output( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        nn->modifyParamWholeLayer( i, learning_rate * pd_loss_wrt_output );
      }
    } else {
      // original
      float current_loss = 0;
      float pd_loss_wrt_output = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        pd_loss_wrt_output += compute_pd_loss_wrt_output( ground_truth_output( 0, i ), nn->output()( 0, i ) );
        current_loss += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      // 4/3 learning rate
      float lr_4_3 = 4.0 / 3 * learning_rate;
      // modifying weights and biases
      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        nn->modifyParamWholeLayer( i, lr_4_3 * pd_loss_wrt_output );
      }
      // compute loss
      nn->apply( input );
      float loss_4_3 = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        loss_4_3 += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      // 2/3 learning rate
      float lr_2_3 = 2.0 / 3 * learning_rate;
      // modifying weights and biases
      for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
        nn->modifyParamWholeLayer( i, ( -lr_4_3 + lr_2_3 ) * pd_loss_wrt_output );
      }
      // compute loss
      nn->apply( input );
      float loss_2_3 = 0;
      for ( int i = 0; i < (int)output_size; i++ ) {
        loss_2_3 += loss_function( ground_truth_output( 0, i ), nn->output()( 0, i ) );
      }

      float min_loss = min( min( current_loss, loss_4_3 ), loss_2_3 );
      if ( min_loss == current_loss ) {
        //  update learning rate
        learning_rate = lr_2_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          nn->modifyParamWholeLayer( i, -lr_2_3 * pd_loss_wrt_output );
        }
      } else if ( min_loss == loss_4_3 ) {
        //  update learning rate
        learning_rate = lr_4_3;
        // update params
        for ( int i = 0; i < (int)nn->getNumLayers(); i++ ) {
          nn->modifyParamWholeLayer( i, ( -lr_2_3 + lr_4_3 ) * pd_loss_wrt_output );
        }
      } else {
        //  no need
      }
    }
  }
};
