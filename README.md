# Neural Network fun

[![Compiles](https://github.com/stanford-stagecast/nnfun/workflows/Compile/badge.svg?event=push)](https://github.com/stanford-stagecast/nnfun/actions)

#### Setting up

0. For best experience, make sure you are on Ubuntu environment.

1. Make sure you have all the following dependencites installed:

   - python3
   - cmake version >= 2.8.5
   - libeigen3-dev

2. Clone this repo.

   `git clone https://github.com/stanford-stagecast/nnfun.git`

3. The rest should be straightforward:

   ```
   cd nnfun/
   mkdir build
   cd build/
   cmake ..
   make -j$16  # make -j$(nproc)
   make check # should pass the 3 tests
   ```

#### How to Run

0. Head to directory `build`.
1. Run `cmake ..`.
2. Run `make`.
3. Run `./src/frontend/<file_to_run>`.

#### File Stucture

- `src`: the directory containing basically all codes for the neural network
  - `nn`: the directory containing files of a generic neural network (please read the file, class, function descriptions in the files)
    - `layer.hh`: atomic element in this neural network
    - `network.hh`: network build using class`Layer`
    - `neuralnetwork.hh`: more higher level class directly providing gradient descent functionality
  - `frontend`: files using the neural network to predict
    - `CMakeLists.txt`: containing files to compile (feel free to comment out some files and add more files)
    - `predict_inverse_256.txt`: the file containing codes to predict 1/x in domain [1,100] (feel free to play with it)
  - `test`: testing files used in `make check` command (feel free to add more)
- `docs`: the directory containing some documentations and thoughts
  - `how_to_use_neuralnetwork.hh.txt`: the txt file containing some useful methods in `neuralnetwork.hh` 
  - `data_structure_ideas.txt.` and `tempo representation.txt`: some previous ideas on note representations
- progress_reports: the directory containing some progress at certain time
  - `06_14_2022.txt`: latest progress before summer 2022