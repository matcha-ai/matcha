cmake_minimum_required(VERSION 3.17)
project(
  matcha-engine VERSION 0.0
  DESCRIPTION "Attachable Machine Learning - Core engine"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pg)

set(
  PROJECT_HEADERS

  include/matcha/tensor
  include/matcha/engine

  include/bits_of_matcha/context.h
  include/bits_of_matcha/object.h
  include/bits_of_matcha/tensor.h
  include/bits_of_matcha/params.h
  include/bits_of_matcha/stream.h
  include/bits_of_matcha/input.h
  include/bits_of_matcha/tuple.h
  include/bits_of_matcha/flow.h
  include/bits_of_matcha/plt.h

  include/bits_of_matcha/engine/object.h
  include/bits_of_matcha/engine/tensor.h
  include/bits_of_matcha/engine/params.h
  include/bits_of_matcha/engine/stream.h
  include/bits_of_matcha/engine/relay.h
  include/bits_of_matcha/engine/input.h
  include/bits_of_matcha/engine/status.h
  include/bits_of_matcha/engine/node.h
  include/bits_of_matcha/engine/in.h
  include/bits_of_matcha/engine/out.h
  include/bits_of_matcha/engine/fn.h
  include/bits_of_matcha/engine/nodeloader.h
  include/bits_of_matcha/engine/nodeserializer.h
  include/bits_of_matcha/engine/flow.h
  include/bits_of_matcha/engine/flowsaver.h
  include/bits_of_matcha/engine/flowloader.h
  include/bits_of_matcha/engine/inlet.h
  include/bits_of_matcha/engine/outlet.h
  include/bits_of_matcha/engine/debug.h

  include/bits_of_matcha/fn/add.h
  include/bits_of_matcha/fn/subtract.h
  include/bits_of_matcha/fn/multiply.h
  include/bits_of_matcha/fn/divide.h

  include/bits_of_matcha/fn/identity.h
  include/bits_of_matcha/fn/abs.h
  include/bits_of_matcha/fn/square.h
  include/bits_of_matcha/fn/sqrt.h
  include/bits_of_matcha/fn/exp.h

  include/bits_of_matcha/fn/matmul.h

  include/bits_of_matcha/fn/transpose.h
  include/bits_of_matcha/fn/reshape.h

  include/bits_of_matcha/fn/min.h
  include/bits_of_matcha/fn/max.h
  include/bits_of_matcha/fn/maxBetween.h
  include/bits_of_matcha/fn/maxIn.h
  include/bits_of_matcha/fn/maxAlong.h

  include/bits_of_matcha/fn/argmin.h
  include/bits_of_matcha/fn/argmax.h
  include/bits_of_matcha/fn/argmaxIn.h

  include/bits_of_matcha/fn/sum.h

  include/bits_of_matcha/fn/equal.h
  include/bits_of_matcha/fn/nequal.h
  include/bits_of_matcha/fn/less.h
  include/bits_of_matcha/fn/greater.h
  include/bits_of_matcha/fn/lequal.h
  include/bits_of_matcha/fn/gequal.h

  include/bits_of_matcha/fn/lnot.h
  include/bits_of_matcha/fn/land.h
  include/bits_of_matcha/fn/lor.h

  include/bits_of_matcha/fn/batch.h
  include/bits_of_matcha/fn/map.h
  include/bits_of_matcha/fn/fold.h

  include/bits_of_matcha/rng/normal.h
  include/bits_of_matcha/rng/uniform.h
  include/bits_of_matcha/rng/poisson.h
  include/bits_of_matcha/rng/bernoulli.h

)

set(
  PROJECT_SOURCES

  src/bits_of_matcha/context.cpp
  src/bits_of_matcha/object.cpp
  src/bits_of_matcha/tensor.cpp
  src/bits_of_matcha/params.cpp
  src/bits_of_matcha/stream.cpp
  src/bits_of_matcha/input.cpp
  src/bits_of_matcha/tuple.cpp
  src/bits_of_matcha/flow.cpp
  src/bits_of_matcha/plt.cpp

  src/bits_of_matcha/engine/object.cpp
  src/bits_of_matcha/engine/tensor.cpp
  src/bits_of_matcha/engine/params.cpp
  src/bits_of_matcha/engine/stream.cpp
  src/bits_of_matcha/engine/relay.cpp
  src/bits_of_matcha/engine/input.cpp
  src/bits_of_matcha/engine/status.cpp
  src/bits_of_matcha/engine/node.cpp
  src/bits_of_matcha/engine/in.cpp
  src/bits_of_matcha/engine/out.cpp
  src/bits_of_matcha/engine/fn.cpp
  src/bits_of_matcha/engine/nodeloader.cpp
  src/bits_of_matcha/engine/nodeserializer.cpp
  src/bits_of_matcha/engine/flow.cpp
  src/bits_of_matcha/engine/flowsaver.cpp
  src/bits_of_matcha/engine/flowloader.cpp
  src/bits_of_matcha/engine/inlet.cpp
  src/bits_of_matcha/engine/outlet.cpp
  src/bits_of_matcha/engine/debug.cpp

  src/bits_of_matcha/fn/add.cpp
  src/bits_of_matcha/fn/subtract.cpp
  src/bits_of_matcha/fn/multiply.cpp
  src/bits_of_matcha/fn/divide.cpp

  src/bits_of_matcha/fn/identity.cpp
  src/bits_of_matcha/fn/abs.cpp
  src/bits_of_matcha/fn/square.cpp
  src/bits_of_matcha/fn/sqrt.cpp
  src/bits_of_matcha/fn/exp.cpp

  src/bits_of_matcha/fn/matmul.cpp

  src/bits_of_matcha/fn/transpose.cpp
  src/bits_of_matcha/fn/reshape.cpp

  src/bits_of_matcha/fn/min.cpp
  src/bits_of_matcha/fn/max.cpp
  src/bits_of_matcha/fn/maxBetween.cpp
  src/bits_of_matcha/fn/maxIn.cpp
  src/bits_of_matcha/fn/maxAlong.cpp

  src/bits_of_matcha/fn/argmin.cpp
  src/bits_of_matcha/fn/argmax.cpp
  src/bits_of_matcha/fn/argmaxIn.cpp

  src/bits_of_matcha/fn/sum.cpp

  src/bits_of_matcha/fn/equal.cpp
  src/bits_of_matcha/fn/nequal.cpp
  src/bits_of_matcha/fn/less.cpp
  src/bits_of_matcha/fn/greater.cpp
  src/bits_of_matcha/fn/lequal.cpp
  src/bits_of_matcha/fn/gequal.cpp

  src/bits_of_matcha/fn/lnot.cpp
  src/bits_of_matcha/fn/land.cpp
  src/bits_of_matcha/fn/lor.cpp

  src/bits_of_matcha/fn/batch.cpp
  src/bits_of_matcha/fn/map.cpp
  src/bits_of_matcha/fn/fold.cpp

  src/bits_of_matcha/rng/normal.cpp
  src/bits_of_matcha/rng/uniform.cpp
  src/bits_of_matcha/rng/poisson.cpp
  src/bits_of_matcha/rng/bernoulli.cpp

)

find_package(matcha-device)
find_package(matcha)
find_package(Threads)
find_package(Boost REQUIRED COMPONENTS serialization)

add_library(${PROJECT_NAME} SHARED)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE 
    include/
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
  ${PROJECT_HEADERS}
  ${PROJECT_SOURCES}
)

add_executable(main src/main.cpp)
#add_executable(benchmark src/benchmark.cpp)

#target_include_directories(
#  ${PROJECT_NAME}
#  PRIVATE
#    /home/patz/Heap/matcha-device/include
#)

target_include_directories(
  main
  PUBLIC
    matcha-device/include
)

target_include_directories(
  main
  PRIVATE
    include/
)

#target_include_directories(
#  benchmark
#  PRIVATE
#    include/
#)

target_link_libraries(${PROJECT_NAME} PUBLIC matcha-device)
target_link_libraries(${PROJECT_NAME} PUBLIC matcha)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::serialization)
target_link_libraries(main PUBLIC ${PROJECT_NAME})
#target_link_libraries(benchmark PUBLIC ${PROJECT_NAME})


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/matcha-engineConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS ${PROJECT_NAME}
    EXPORT matcha-engineTargets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/matcha-engineConfig.cmake"
    "${PROJECT_BINARY_DIR}/matcha-engineConfig.cmake"
    INSTALL_DESTINATION lib/cmake/matcha-engine
)

install(EXPORT matcha-engineTargets DESTINATION lib/cmake/matcha-engine)
install(FILES "${PROJECT_BINARY_DIR}/matcha-engineConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/matcha-engineConfig.cmake"
        DESTINATION lib/cmake/matcha-engine)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
