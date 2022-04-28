#include "exception.hh"
#include "network.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <utility>

#include <sys/resource.h>

using namespace std;
using namespace Eigen;

float loss_function( const float target, const float actual ) {
  return (target - actual) * (target - actual);
}

float compute_pd_loss_wrt_output( const float target, const float actual ) {
  return -2 * (target - actual);
}

float true_function( const float input ) {
  return 1.0 / input;
}

float learning_rate = 0.001;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 1;
constexpr size_t output_size = 1;
vector<unsigned int> layers{ input_size, 1, output_size };

void program_body() {
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 1, output_size>>();
  auto temp = *(&nn);
  vector<Network> pointer_to_layers;
  for (int i = 0; i < layers.size(); i++) {
    cout << "hi" << endl; 
  }

  //nn->print();
}

int main( int argc, char*[] ) {
  try {
    if ( argc <= 0 ) {abort();}
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
      return EXIT_FAILURE;
    }
}
