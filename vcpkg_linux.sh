#!/bin/bash
set -e

mkdir -p build
cd build
git clone https://github.com/Microsoft/vcpkg.git vcpkg.linux
cd vcpkg.linux
git checkout 2019.10
./bootstrap-vcpkg.sh

./vcpkg install \
	boost-program-options:x64-linux \
	glfw3:x64-linux \
