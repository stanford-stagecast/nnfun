/**
  This file creates a one-layer neural network to calculate the beat from 16
  inputs.

  We want range of tempo: 35 - 250 bpm //TODO
*/
#include "exception.hh"
#include "midi_processor.hh"
#include "network.hh"
#include "timer.hh"

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <iostream>
#include <queue>
#include <random>
#include <utility>

#include "eventloop.hh"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>

using namespace std;
using namespace Eigen;

constexpr size_t batch_size = 1;
constexpr size_t input_size = 16;
// constexpr long time_offset = 1653036000;

/* use squared error as loss function */
float loss_function( const float target, const float actual )
{
  return ( target - actual ) * ( target - actual );
}

/* partial derivative of loss with respect to neural network output */
float compute_pd_loss_wrt_output( const float target, const float actual )
{
  return -2 * ( target - actual );
}

/* compute input */
Matrix<float, batch_size, input_size> gen_time( float spb, float offset )
{
  Matrix<float, batch_size, input_size> ret_mat;
  for ( auto i = 0; i < 16; i++ ) {
    ret_mat( 15 - i ) = (spb)*i + offset;
  }
  // cout << ret_mat << endl;
  return ret_mat;
}

/* compute input */
Matrix<float, batch_size, input_size> calculate_input( deque<float> times, int num_notes )
{
  auto current_time = std::chrono::system_clock::now();
  auto duration_in_seconds = std::chrono::duration<double>( current_time.time_since_epoch() );

  float curr_time_secs = duration_in_seconds.count();

  Matrix<float, batch_size, input_size> ret_mat;
  for ( auto i = 0; i < 16; i++ ) {
    if ( i >= num_notes ) {
      ret_mat( i ) = 0;
    } else {
      ret_mat( i ) = curr_time_secs - times[i];
    }
  }
  // cout << ret_mat << endl;
  return ret_mat;
}

float learning_rate = 0.0001;

void program_body( const string& midi_filename )
{
  /* remove limit on stack size */
  const rlimit limits { RLIM_INFINITY, RLIM_INFINITY };
  CheckSystemCall( "setrlimit", setrlimit( RLIMIT_STACK, &limits ) );

  /* seed C RNG for Eigen random weight initialization */
  srand( Timer::timestamp_ns() );

  /* construct neural network on heap */
  auto nn = make_unique<Network<float, batch_size, input_size, 1>>();
  nn->layer0.initializeWeightsRandomly();
  /*for ( int i = 0; i < 16; i++ ) {
    nn->layer0.weights()( i ) = 0;
  }
  nn->layer0.biases()( 0 ) = 0;
  */
  float offset = 0;
  for ( int run = 0; run < 2000; run++ ) {
    for ( int tempo_int = 20; tempo_int < 200; tempo_int++ ) {
      /* test true function */
      // float tempo_int_perturb = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/4));
      // tempo = 60.0/(tempo_int + tempo_int_perturb);
      float spb = 60.0 / tempo_int;
      int i = 0;
      while ( true ) {
        if ( i == 1 )
          break;
        i += 1;
        // float rand_offset = 0;
        float rand_offset = static_cast<float>( rand() ) / ( static_cast<float>( RAND_MAX / 5 ) );
        /* step 1: construct a unique problem instance */
        Matrix<float, batch_size, input_size> input = gen_time( spb, offset + rand_offset );

        /* step 2: forward propagate and calculate loss functiom */
        nn->apply( input );
        // cout << "nn maps input: tempo: " << tempo << " offset " << offset + rand_offset << " => "
        //      << nn->output()( 0, 0 ) << endl;

        /* step 3: backpropagate error */
        nn->computeDeltas();
        nn->evaluateGradients( input );

        const float pd_loss_wrt_output = compute_pd_loss_wrt_output( spb, nn->output()( 0, 0 ) );

        // TODO: static eta -> dynamic eta
        auto four_third_lr = 4.0 / 3 * learning_rate;
        auto two_third_lr = 2.0 / 3 * learning_rate;

        /* calculate three loss */
        float current_loss = loss_function( nn->output()( 0, 0 ), spb );
        Matrix<float, input_size, 1> current_weights;
        for ( int j = 0; j < 16; j++ ) {
          current_weights( j ) = nn->layer0.weights()( j );
        }
        auto current_biase = nn->layer0.biases()( 0 );

        /* loss for 4/3 eta */
        for ( int j = 0; j < 16; j++ ) {
          nn->layer0.weights()( j ) -= four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
        }
        nn->layer0.biases()( 0 ) -= four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
        nn->apply( input );
        auto loss_four_third_lr = loss_function( nn->output()( 0, 0 ), spb );

        /* loss for 2/3 eta */
        for ( int j = 0; j < 16; j++ ) {
          nn->layer0.weights()( j )
            = current_weights( j ) - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
        }
        nn->layer0.biases()( 0 )
          = current_biase - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
        nn->apply( input );
        auto loss_two_third_lr = loss_function( nn->output()( 0, 0 ), spb );

        // cout << current_loss << " " << loss_four_third_lr << " " << loss_two_third_lr << endl;
        auto min_loss = min( min( current_loss, loss_four_third_lr ), loss_two_third_lr );
        if ( min_loss == current_loss ) {
          learning_rate *= 2.0 / 3;
          for ( int j = 0; j < 16; j++ ) {
            nn->layer0.weights()( j ) = current_weights( j );
          }
          nn->layer0.biases()( 0 ) = current_biase;
        } else if ( min_loss == loss_four_third_lr ) {
          learning_rate *= 4.0 / 3;
          for ( int j = 0; j < 16; j++ ) {
            nn->layer0.weights()( j )
              = current_weights( j ) - four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
          }
          nn->layer0.biases()( 0 )
            = current_biase - four_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
        } else {
          for ( int j = 0; j < 16; j++ ) {
            nn->layer0.weights()( j )
              = current_weights( j ) - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, j );
          }
          nn->layer0.biases()( 0 )
            = current_biase - two_third_lr * pd_loss_wrt_output * nn->getEvaluatedGradient( 0, 16 );
        }
        // cout << "weights: " << nn->layer0.weights() << endl;
        // cout << "biase: " << nn->layer0.biases()( 0 ) << endl;
      }
    }
  }

  FileDescriptor piano { CheckSystemCall( midi_filename, open( midi_filename.c_str(), O_RDONLY ) ) };
  MidiProcessor midi_processor {};
  midi_processor.reset_time();
  auto event_loop = make_shared<EventLoop>();
  deque<float> press_queue {};
  int num_notes = 0;

  Matrix<float, batch_size, input_size> ret_mat;

  /* rule #1: read events from MIDI piano */
  event_loop->add_rule( "read MIDI data", piano, Direction::In, [&] { midi_processor.read_from_fd( piano ); } );

  /* rule #2: add MIDI data to matrix */
  event_loop->add_rule(
    "synthesizer processes data",
    [&] {
      while ( midi_processor.has_event() ) {
        uint8_t event_type = midi_processor.get_event_type();
        float time_val = midi_processor.pop_event() / 1000.0;
        if ( event_type == 144 ) {
          if ( num_notes < 16 ) {
            press_queue.push_back( time_val );
            cout << "time val: " << time_val << "\n";
            num_notes++;
          } else {
            press_queue.push_back( time_val );
            press_queue.pop_front();
          }
          ret_mat = calculate_input( press_queue, num_notes );
          nn->apply( ret_mat );
          cout << "prediction: " << 60 / ( nn->output()( 0, 0 ) ) << endl;
        }
      }
    },
    /* when should this rule run? */
    [&] { return midi_processor.has_event(); } );

  while ( event_loop->wait_next_event( 5 ) != EventLoop::Result::Exit ) {
  }
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      return EXIT_FAILURE;
    }

    program_body( argv[1] );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}