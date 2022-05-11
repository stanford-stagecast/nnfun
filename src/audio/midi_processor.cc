#include "midi_processor.hh"

using namespace std;
using namespace chrono;

void MidiProcessor::read_from_fd( FileDescriptor& fd )
{
  unprocessed_midi_bytes_.push_from_fd( fd );

  pop_active_sense_bytes();
}
