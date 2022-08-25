/**
  This file uses the successfuly leaky nn to predict tempo from live inputs on the piano
  Last edited: july 26, 2022
*/
#include "exception.hh"
#include "midi_processor.hh"
#include "timer.hh"
#include "grant_neuralnetwork.hh"

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
constexpr size_t output_size = 1;
constexpr size_t num_layers = 5;
constexpr size_t layer_size1 = 16;
constexpr size_t layer_size2 = 1;
constexpr size_t layer_size3 = 30;
constexpr size_t layer_size4 = 2560;
constexpr float learning_rate = 0.000000001;

// float CURR_TIME;

float get_current_time(std::chrono::steady_clock::time_point base_time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - base_time ).count()/1000.0;
}
/* Create Input to NN -- Matrix of 16 timestamps (Notes) */
Matrix<float, batch_size, input_size> calculate_input( deque<float> times, int num_notes, std::chrono::steady_clock::time_point base_time )
{
  float curr_time_secs = get_current_time(base_time);
  // CURR_TIME = curr_time_secs;

  Matrix<float, batch_size, input_size> ret_mat;
  for ( size_t i = 0; i < input_size; i++ ) {
    if ( i >= size_t(num_notes) ) {
      ret_mat( i ) = 0;
    } else {
      ret_mat( i ) = curr_time_secs - times[i];
    }
  }
  //cout << "Timestamps: {" << ret_mat << "}" << endl;
  // cout << "Current time: " << curr_time_secs << endl;
  return ret_mat;
}

/* Run NN using piano -- live tempo prediction */
void nn_input_from_piano( const string& midi_filename) {
  
  /* Initialize Neural Network */
  auto nn = make_unique<NeuralNetwork<float, num_layers, batch_size, input_size, output_size, layer_size1, layer_size2, layer_size3, layer_size4, output_size>>();
  nn->initialize(learning_rate);
  string weights_file = "/home/gbishko/nnfun/src/frontend/grant_trial_weights.txt";
  nn->init_params(weights_file);

  /* Set Up Piano */
  FileDescriptor piano { CheckSystemCall( midi_filename, open( midi_filename.c_str(), O_RDONLY ) ) };
  MidiProcessor midi_processor {};
  midi_processor.reset_time();
  std::chrono::steady_clock::time_point base_time = midi_processor.get_original_time();
  auto event_loop = make_shared<EventLoop>();
  deque<float> press_queue {};
  int num_notes = 0;
  int num_pred_notes = 0;

  Matrix<float, batch_size, input_size> actual_mat;
  deque<float> predicted_queue {};
  Matrix<float, batch_size, input_size> pred_mat;
  float time_val;

  /* rule #1: read events from MIDI piano */
  event_loop->add_rule( "read MIDI data", piano, Direction::In, [&] { midi_processor.read_from_fd( piano ); } );
  /* rule #2: add MIDI data to matrix */
  event_loop->add_rule(
      "synthesizer processes data",
      [&] {
      while ( midi_processor.has_event() ) {
          uint8_t event_type = midi_processor.get_event_type();
          time_val = midi_processor.pop_event() / 1.0;
          if ( event_type == 144 ) {
            if ( num_notes < 16 ) {
                press_queue.push_front( time_val );
                num_notes++;
            } else {
                press_queue.push_front( time_val );
                press_queue.pop_back();
            }
          }
      }
      },
      [&] { return midi_processor.has_event(); } );

  /* Apply Neural Network */
  while ( event_loop->wait_next_event( 50 ) != EventLoop::Result::Exit ) {
    bool PLAY_NOTE = true;
    actual_mat = calculate_input( press_queue, num_notes, base_time );
    nn->apply_leaky( actual_mat );
    float next_timestamp = nn->get_output()( 0, 0 );   // in how many seconds the next note should be played (-)
    float most_recent_timestamp = actual_mat[0];          
    float seconds_left = most_recent_timestamp + next_timestamp;    // how many seconds left until the next note should be played
    
    /* Predicted time is NOW (Play the note!) */
    if ( seconds_left >= -0.05 && seconds_left <= 0.05 && PLAY_NOTE) {
      float curr_time_secs = get_current_time(base_time);
      cout << "\nPLAY NOTE NOW\n" << curr_time_secs << endl;
      if ( predicted_queue.size() < 16 ) {
          predicted_queue.push_front( curr_time_secs );
          num_pred_notes++;
      } else {
          predicted_queue.push_front( curr_time_secs );
          predicted_queue.pop_back();
      }
      // cout << "Predicted {";
      // for (float elem : predicted_queue) cout << elem << " ";
      // cout << "}" << endl;

      PLAY_NOTE = false;

      cout << "Actual: {" << actual_mat << "}" << endl;
      cout << "Predicted: {" << pred_mat << "}" << endl;

    } else {
      PLAY_NOTE = true;
    }
    pred_mat = calculate_input(predicted_queue, num_pred_notes, base_time);


    // for ( size_t i = 0; i < input_size; i++ ) {
    //   if ( i >= size_t(num_notes) ) {
    //     pred_mat( i ) = 0;
    //   } else {
    //     pred_mat( i ) = CURR_TIME - predicted_queue[i];
    //   }
    // }

  }
}


int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      cout << "No keyboard selected" << endl;
      return EXIT_FAILURE;
    }

    nn_input_from_piano(argv[1]);

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}