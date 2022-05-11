#pragma once

#include <chrono>
#include <optional>

#include "file_descriptor.hh"
#include "ring_buffer.hh"

/* wrap MIDI file input */
class MidiProcessor
{
  RingBuffer unprocessed_midi_bytes_ { 4096 };

  std::chrono::steady_clock::time_point last_event_time_ { std::chrono::steady_clock::now() };

public:
  void read_from_fd( FileDescriptor& fd );

  unsigned int pop_event()
  {
    while ( unprocessed_midi_bytes_.readable_region().size() >= 3 ) {
      unprocessed_midi_bytes_.pop( 3 );
      pop_active_sense_bytes();
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now()
                                                                  - last_event_time_ )
      .count();
  };

  void pop_active_sense_bytes()
  {
    while ( unprocessed_midi_bytes_.readable_region().size()
            and uint8_t( unprocessed_midi_bytes_.readable_region().at( 0 ) ) == 0xfe ) {
      unprocessed_midi_bytes_.pop( 1 );
    }
  }

  bool want_read() const { return unprocessed_midi_bytes_.writable_region().size() > 0; }

  bool has_event() { return unprocessed_midi_bytes_.readable_region().size() >= 3; }

  uint8_t get_event_type() const { return unprocessed_midi_bytes_.readable_region().at( 0 ); }
  uint8_t get_event_note() const { return unprocessed_midi_bytes_.readable_region().at( 1 ); }
  uint8_t get_event_velocity() const { return unprocessed_midi_bytes_.readable_region().at( 2 ); }

  // unsigned int pop_event();

  void reset_time() { last_event_time_ = std::chrono::steady_clock::now(); };
};
