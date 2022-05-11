#pragma once

#include "network.hh"

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;


class NeuralNetwork
{
private:
  unsigned int num_of_layers; // total numebr of layers (excluding output)
                              // 1 -> 2 -> 4 -> 1     3 layers
  unsigned int input_size;    // number of inputs to the nn
  unsigned int output_size;   // number of outputs from the nn
  unsigned int batch_size;
  class T;
  Network<T, batch_size, input_size, rest...> network;

public:
  // ctor
  NeuralNetwork(class T, unsigned int b_size, unsigned int num_layers, 
                vector<unsigned int> layers) {
    cout << layers.size() << endl;
    num_of_layers = num_layers;
    batch_size = b_size;
    input_size = layers.front();
    output_size = layers.back();
    network = make_unique<Network<float, batch_size, layers.front(), layers.back()>>();
  }

  // getters
  unsigned int get_num_of_layers() {
    return num_of_layers;
  }

  unsigned int get_input_size() {
    return input_size;
  }

  unsigned int get_output_size() {
    return output_size;
  }
};
