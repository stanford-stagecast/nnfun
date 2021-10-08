#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "timer.hh"


using namespace std;
using namespace Eigen;

typedef MatrixXf Matrix;
typedef RowVectorXf RowVector;
typedef VectorXf ColVector;

class NeuralNetwork {
    vector<ColVector*> neuronLayers; // different layers of output network
    vector<ColVector*> cacheLayers; // values of layers before activation
    vector<ColVector*> deltas; // stores the error contribution of each neurons
    vector<MatrixXf*> weights; // the weights of connections between layers
    vector<uint> topology;
    float learningRate;
    // enum { NeedsToAlign = (sizeof(neuronLayers)%16)==0 };

  public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

    NeuralNetwork(vector<uint> topology, float learningRate = 0.005f);

    ColVector forward_propagation(ColVector& input);
    void backward_propagation(ColVector& output);
    void error_calculation(ColVector& output);
    void update_weights();

    float activationFunction(float x){
      return max(x, 0.0f);
    }

    ColVector activationFunction(ColVector& x){
      return x.cwiseMax(0);
    }

    float activationFunctionDerivative(float x){
      if (x >= 0)
        return 1;
      else
        return 0;
    }

    void print(){
      const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );
      for(uint i = 1; i < topology.size(); i++){
        cout << "Weights Layer " << i << ":\n"
           << (*weights[i - 1]).format( CleanFmt ) << "\n\n"
           << "cacheLayers: Layer " << i << ":\n"
           << (*cacheLayers[i]).format( CleanFmt ) << "\n\n"
           << "neuronLayers Layer " << i << ":\n"
           << (*neuronLayers[i]).format( CleanFmt ) << "\n\n";
      }
    }
};


NeuralNetwork::NeuralNetwork(vector<uint> topology_, float learningRate_): neuronLayers(), cacheLayers(), deltas(), weights(), topology(topology_), learningRate(learningRate_){
  topology = topology_;
  learningRate = learningRate_;
  for (uint i = 0; i < topology.size(); i++) {
    if (i != topology.size() - 1)
      neuronLayers.push_back(new ColVector(topology[i] + 1));
      // +1 for biases
    else
      neuronLayers.push_back(new ColVector(topology[i]));

    // initialize cache and delta vectors
    cacheLayers.push_back(new ColVector(neuronLayers[i]->rows()));
    deltas.push_back(new ColVector(neuronLayers[i]->rows()));

    // setting up biases nodes
    if (i != topology.size() - 1) {
      neuronLayers.back()->coeffRef(topology[i]) = 1.0;
      cacheLayers.back()->coeffRef(topology[i]) = 1.0;
    }

    // initialize weights matrix
    if (i > 0) {
      if (i != topology.size() - 1) {
        weights.push_back(new MatrixXf(topology[i] + 1, topology[i - 1] + 1));

        // random weights initialisation
        weights.back()->setRandom();

        // no "normal" weights into biases
        weights.back()->row(topology[i]).setZero();
        weights.back()->coeffRef(topology[i], topology[i - 1]) = 1.0;
      }
      else {
        weights.push_back(new MatrixXf(topology[i], topology[i - 1] + 1));
        weights.back()->setRandom();
      }
    }
  }
}

ColVector NeuralNetwork::forward_propagation(ColVector& input)
{
  // setting the input layer of the NN
  // block takes 4 arguments : startRow, startCol, blockRows, blockCols

  neuronLayers.front()->block(0, 0, neuronLayers.front()->size() - 1, 1) = input;

  // forward propagation
  for (uint i = 1; i < topology.size() - 1; i++) {
    (*cacheLayers[i]) = (*weights[i - 1]) * (*neuronLayers[i - 1]);
    (*neuronLayers[i]) = activationFunction(*cacheLayers[i]);
    //resetting back the bias NN node
    neuronLayers[i]->coeffRef(topology[i]) = 1.0;
  }
  // no activation at last layer
  uint i = topology.size() - 1;
  (*cacheLayers[i]) = (*weights[i - 1]) * (*neuronLayers[i - 1]);
  (*neuronLayers[i]) = *cacheLayers[i];

  return *neuronLayers[topology.size()-1];
}

void program_body()
{
  vector<uint> TOPOLOGY = {5, 3, 2, 1};
  ColVector input(5);
  input << 1.0f, -2.0f, 3.0f, -2.0f, 0.0f;
  NeuralNetwork nn(TOPOLOGY);
  uint64_t start = Timer::timestamp_ns();
  ColVector output = nn.forward_propagation(input);
  uint64_t end = Timer::timestamp_ns();
  cout << end - start << endl;

  const IOFormat CleanFmt( 4, 0, ", ", "\n", "[", "]" );

  cout << "input:\n"
       << input.format( CleanFmt ) << "\n\n"
       << "output:\n"
       << output.format( CleanFmt ) << "\n\n";
  nn.print();

}

int main()
{
  try {
    ios::sync_with_stdio( false );
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
