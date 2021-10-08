#pragma once

#include "layer.hh"

template<unsigned int... sizes>
class Network
{
  /* todo: Network should include a series of Layers where
     the output size of each Layer matches the input size
     of the next layer... */

public:
  void forward_pass() {}
};
