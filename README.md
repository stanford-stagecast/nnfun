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

