#!/bin/sh
# Build this fork on a machine without root and without the clang / fontconfig
# development packages (only their runtime libraries):
#   - fontconfig is loaded at runtime instead of linked (no fontconfig.pc needed)
#   - bindgen uses the libclang runtime, with GCC's builtin headers (stddef.h …)
#     because the clang resource directory is not installed
# Usage: ./build.sh [extra cargo args]   -> target/release/bookokrat
set -eu
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
export RUST_FONTCONFIG_DLOPEN=on
if [ -z "${LIBCLANG_PATH:-}" ]; then
  for d in /usr/lib/llvm-*/lib; do [ -e "$d/libclang.so" ] || ls "$d"/libclang-*.so* >/dev/null 2>&1 && LIBCLANG_PATH=$d; done
  export LIBCLANG_PATH
fi
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]; then
  gccinc=$(dirname "$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include/stddef.h 2>/dev/null | sort -V | tail -1)")
  [ -n "$gccinc" ] && export BINDGEN_EXTRA_CLANG_ARGS="-I$gccinc"
fi
exec cargo build --release "$@"
