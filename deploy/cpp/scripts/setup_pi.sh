#!/usr/bin/env bash
# Cài thư viện và biên dịch chương trình C++ trên Raspberry Pi OS 64-bit (Bookworm).
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.20.1}"
cd "$(dirname "$0")/.."

if [ "${SKIP_APT:-0}" != "1" ]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake wget libopencv-dev alsa-utils
fi

case "$(uname -m)" in
  aarch64) ORT_ARCH=aarch64 ;;
  x86_64)  ORT_ARCH=x64 ;;
  *) echo "Kiến trúc $(uname -m) không được hỗ trợ (cần Raspberry Pi OS 64-bit)"; exit 1 ;;
esac

if [ ! -d third_party/onnxruntime ]; then
  mkdir -p third_party
  pkg="onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}"
  wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${pkg}.tgz" -O "/tmp/${pkg}.tgz"
  tar -xzf "/tmp/${pkg}.tgz" -C third_party
  mv "third_party/${pkg}" third_party/onnxruntime
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "Xong: ./build/tsr --help"
