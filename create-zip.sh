#!/bin/bash -e

set -e
scriptdir="$( cd "$(dirname "$0")" ; pwd -P )"

cp ../Meadow/nuttx/Meadow.OS.* .
zip ../Tensorflow.zip Tensorflow.so Meadow.OS.*
rm Meadow.OS.*
