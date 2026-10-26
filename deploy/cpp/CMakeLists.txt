cmake_minimum_required(VERSION 3.16)
project(traffic_sign_recognition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Thư mục ONNX Runtime đã giải nén (chứa include/ và lib/), xem docs/deploy_raspberry_pi.md
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)
find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(tsr src/main.cpp src/detector.cpp)
target_include_directories(tsr PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tsr PRIVATE ${ORT_LIB} ${OpenCV_LIBS} Threads::Threads)
set_target_properties(tsr PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib" INSTALL_RPATH "${ORT_ROOT}/lib")
