# Tensorflow

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
| |____Tensorflow repositry...
```

This is necessary are the Tensorflow library requires the `libbuiltin.a` archive file in the `Meadow` repository.

The `Tensorflow.so` DLL can be built by running the `build.sh` script located in the root of the repository.

## Licence

The original license in the Spresense repository is a BSD-3-Clause

* [Original license](https://github.com/sonydevworld/spresense/blob/master/LICENSE)
