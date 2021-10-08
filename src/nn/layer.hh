#pragma once

#include <array>

template<unsigned int input_size, unsigned int output_size>
class Layer
{
  std::array<float, output_size> activations_;

public:
  void apply( const std::array<float, input_size>& input __attribute( ( unused ) ) ) {}

  const std::array<float, output_size>& activations() const { return activations_; }
};
