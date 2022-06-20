# Neural Network fun

[![Compiles](https://github.com/stanford-stagecast/nnfun/workflows/Compile/badge.svg?event=push)](https://github.com/stanford-stagecast/nnfun/actions)

#### Setting up

0. For best experience, make sure you are on Ubuntu environment.

1. Make sure you have all the following dependencites installed:

   - python3
   - cmake version >= 2.8.5
   - libeigen3-dev

2. Clone this repo.

   `git clone https://github.com/stanford-stagecast/nnfun.git `

3. The rest should be straightforward:

   ```
   cd nnfun/
   mkdir build
   cd build/
   cmake ..
   make -j$16  # make -j$(nproc)
   make check # should pass the 3 tests
   ```

## Testing continuous midi prediction

1. ssh [your-name]@snr-piano.stanford.edu

2. Clone this repo and run 

   git checkout any_16_input_nn

to get to the correct branch.

3. Build/make as shown in step 3 above.

4. In the /build directory, run ./src/frontend/predict_tempo_midi_continuous /dev/midi2

5. Press the piano keys to receive a live prediction of the tempo from your last 16 note inputs.
