#include "neuralnetwork.hh"

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>

#include "timer.hh"

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 16;
constexpr size_t output_size = 1;
constexpr size_t num_layers = 5;
constexpr size_t layer_size1 = 16;
constexpr size_t layer_size2 = 1;
constexpr size_t layer_size3 = 30;
constexpr size_t layer_size4 = 2560;

float get_rand(float min, float max)
{
  return (rand()/(RAND_MAX + 1.0 ))*(max-min) + min;
}

Matrix<float, batch_size, input_size> gen_time( float tempo, bool offset, bool noise )
{
  Matrix<float, batch_size, input_size> ret_mat; //empty matrix [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  float amt_offset = 0;
  if (offset) {
    float sbb  = (60.0 / tempo);   // seconds between beats
    amt_offset = get_rand(0.0, sbb);
  }
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( i ) = (60.0/tempo) * i + offset; //[0, 0.25, 0.5, 0.75, 1, 1.25, ...] (backward in time)
    if (noise) {
      float pct_noise = get_rand(-0.05, 0.05);
      float amt_noise = pct_noise * (60.0/tempo);
      ret_mat(i) += amt_noise;
    }
    ret_mat(i) += amt_offset;
  }
  return ret_mat;
}

void program_body()
{
  auto nn = make_unique<
    NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size1, layer_size2, layer_size3, layer_size4, output_size>>();
  nn->initialize(0.000000001);

  Matrix<float, batch_size, input_size> input;
  Matrix<float, batch_size, output_size> ground_truth_output;
  // input = Matrix<float, batch_size, input_size>::Random();
  //input << 1;
  //ground_truth_output << 60.0 / 1;
  // nn->apply( input );
  // cout << nn->get_output()( 0, 0 ) << endl;
  // const uint64_t b_start = Timer::timestamp_ns();


 /******************** TRAINING ********************/
  bool offset = true;
  bool noise = true;
  int num_updates = 10;
  bool verbose_updates = true;
  float tempo = 0;
  int iterations = 1000000;

  cout << "Beginning training! You will receive " << num_updates << " updates." << endl << endl; 

  for ( int i = 0; i < iterations; i++ ) {
    float prob = get_rand(0, 1);
    if(prob < 0.4)
    {
        tempo = get_rand(30, 50);

    }
    else{
        tempo = get_rand(30, 240);
    }
    input = gen_time( tempo, offset, noise );


    // Train the neural network
    ground_truth_output(0,0) = tempo;
    nn->apply_leaky(input);
    nn->leaky_gradient_descent( input, ground_truth_output, true );

    if(i % (iterations/num_updates) == 0 && i > 0)
    {
      cout << "latest tempo: " << tempo << " bpm" << endl;
      cout << "after " << (float(i)/iterations)*100 << " percent of training:" << endl;

      if(verbose_updates){
        for (  int tempo_test = 30; tempo_test < 241; tempo_test++ ) {
          input = gen_time( tempo_test, offset, noise );
          nn->apply_leaky(input);
          cout << tempo_test << " -> " << nn->get_output()( 0, 0 ) << ", ";
          if(tempo_test % 10 == 9) { cout << endl;}
        }
        cout << endl;
      }
      else{
        float average_diff_of_ten = 0;
        float average_diff = 0;
        float diff;
        for (  int tempo_test = 30; tempo_test < 241; tempo_test++ ) {
          input = gen_time( tempo_test, offset, noise );
          nn->apply_leaky(input);
          diff = abs(tempo_test - nn->get_output()(0, 0));
          average_diff_of_ten += abs(tempo_test - nn->get_output()(0, 0));
          average_diff += diff;
          if(tempo_test % 10 == 9) {
            cout << "Avg diff on " << tempo_test - 9 << "s: " << average_diff_of_ten/10.0 << endl;
            average_diff_of_ten = 0;
          }
        }
        cout << "Overall average diff: " << average_diff/211.0 << endl;      
      }
      cout << endl;

    }
    
  }
  
  cout << endl;
  cout << "******* TRAINING COMPLETE :) *******" << endl;
  cout << endl;
  nn->printWeights("trial_weights.txt");
  cout << "Testing neural network now..." << endl;
  cout << endl;

  // const uint64_t b_end = Timer::timestamp_ns();
  // cout << "timer: " << ( b_end - b_start ) << endl;

  // nn->print();
 /******************** TESTING ********************/
  for (  int tempo_test = 30; tempo_test < 241; tempo_test++ ) {
    input = gen_time( tempo_test, offset, noise );
    // input( 0, 0 ) = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX ) ) * 1.75 + 0.25;
    // ground_truth_output( 0, 0 ) = 60.0 / input( 0, 0 );
    nn->apply_leaky(input);
    cout << tempo_test << " -> " << nn->get_output()( 0, 0 ) << endl;
  }
  cout << endl;
  cout << "Number of layers: " << num_layers << endl;
  cout << "With noise? " << noise << endl;
  cout << "With offset? " << offset << endl;
  cout << "Number of Iterations: " << iterations << endl;
  cout << "Learning Rate: " << nn->get_current_learning_rate() << endl;

}

int main( int argc, char*[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    return EXIT_FAILURE;
  }
}