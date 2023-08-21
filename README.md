# Tensorflow

Build status: [![Main Branch](https://github.com/WildernessLabs/Tensorflow/actions/workflows/build.yml/badge.svg)](https://github.com/WildernessLabs/Tensorflow/actions/workflows/build.yml)

This library is a port of the Sony Spresense Tensorflow port and the original cade can be found here:

* [Main Spresense repository](https://github.com/sonydevworld/spresense)
* [Tensorflow library code](https://github.com/sonydevworld/spresense/tree/master/externals/tensorflow)


## Building the Library

The Tensorflow repository should be cloned at the same level as the Meadow repository, something like this:

```
.
|____Meadow
| |____Meadow repository...
|____Tensorflow
| |____Tensorflow repository...
```

This is necessary for two main reasons:

* The Tensorflow library requires the `libbuiltin.a` archive file in the `Meadow` repository
* Configuration for the Tensorflow library is dependent upon the `.config` file in the Nuttx repository

The `Tensorflow.so` DLL can be built by running the `build.sh` script located in the root of the repository.

## Operating System Unit Test

The first stage to building the OS level unit tests is to turn on the Tensorflow unit tests in NuttX.  To do this, change to the directory containing the NuttX repository and execute the following command:

```bash
./build.sh -u=tensorflow
```

This should build NuttX and include the unit test code into the operating system.  The resulting OS files should be written to the development board as normal.

Next, return to the Tensorflow repository and build the system using the `build.sh` script.  This will create the `Tensorflow.so` and this file should be written to the development board with the command:

```bash
meadow file write -f Tensorflow.so
```
The OS level unit test can be run using the command:

```bash
meadow set developer -d 11
```

The results of the test can be observed by attaching a terminal to the Meadow debug console (COM1) and enabling trace output.  If all has gone well something like the following should be seen:

```
[   43.340000] [ 8] Executing Tensorflow Hello World test
[   43.340000] [ 8]     Opening Tensorflow.so DLL
[   43.732000] [ 8]     Handle: 0x20077640
[   43.733000] [ 8]     tensorflow_hello_world_test_setup: 0xc03923a9
[   43.734000] [ 8]     tensorflow_hello_world_test_loop: 0xc0392535
[   43.747000] [ 8]     Tensorflow Hello World test - PASS.
```
### `tensorflow_hello_world_test_setup` Cannot be Found

Error output such as the following indicates that the OS has been compiled and deployed with Tensorflow unit tests enabled but the `Tensorflow.so` file does not contain the tests:

```
[ 4262.332000] [ 8] Executing Tensorflow Hello World test
[ 4262.332000] [ 8]     Opening Tensorflow.so DLL
[ 4262.779000] [ 8]     Handle: 0x20077640
[ 4262.779000] [ 8]     tensorflow_hello_world_test_setup: 0
[ 4262.779000] [ 8]     Cannot locate the tensorflow_hello_world_test_setup method.
[ 4262.791000] [ 8]     Tensorflow Hello World test - FAIL.
```
#### Resolution:

Reconfigure NuttX with unit tests enabled and then rebuild and redeploy the `Tensorflow.so` file.

### `meadow` Command Show No Developer Tests Found

This indicates that the OS does not have the Tesnorflow unit tests enabled in the OS.

#### Resolution:

Rebuild and redeploy the OS with Tensorflow unit tests enabled.

## Licence

The original license in the Spresense repository is a BSD-3-Clause

* [Original license](https://github.com/sonydevworld/spresense/blob/master/LICENSE)
