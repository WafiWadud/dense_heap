#!/bin/sh

# Check if user passed in wasm as $1, if so, use wasm build command, else normal.
if [ "$1" = "wasm" ]; then
  echo "Building for WebAssembly..."
  set -x
  clang -Wall -Wextra -O3 -fdefer-ts -flto -ffast-math -fno-ident -fno-asynchronous-unwind-tables -fno-stack-protector -funroll-loops -fomit-frame-pointer -ffunction-sections -fdata-sections -fno-exceptions -Wl,--gc-sections,--strip-all --target=wasm32-unknown-wasip1 --sysroot=/usr/share/wasi-sysroot dense_heap.c dense_heap_test.c -o dense_heap_test.wasm -D_WASI_EMULATED_MMAN -lwasi-emulated-mman
else
  echo "Building for native..."
  set -x
  clang -Wall -Wextra -O3 -fdefer-ts -mtune=native -flto -ffast-math -fno-ident -fno-asynchronous-unwind-tables -fno-stack-protector -funroll-loops -fomit-frame-pointer -ffunction-sections -fdata-sections -fno-exceptions -Wl,--gc-sections,--strip-all dense_heap.c dense_heap_test.c -o dense_heap_test
fi
