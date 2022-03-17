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

  include/matcha/matcha
  include/matcha/tensor
  include/matcha/engine
  include/matcha/dataset
  include/matcha/nn

  include/bits_of_matcha/error/error.h

  include/bits_of_matcha/Dtype.h
  include/bits_of_matcha/Shape.h
  include/bits_of_matcha/Frame.h
  include/bits_of_matcha/tensor.h
  include/bits_of_matcha/Generator.h
  include/bits_of_matcha/fn.h
  include/bits_of_matcha/initializable_fn.h
  include/bits_of_matcha/Sequence.h
  include/bits_of_matcha/Flow.h
  include/bits_of_matcha/Grad.h
  include/bits_of_matcha/Slice.h
  include/bits_of_matcha/Device.h
  include/bits_of_matcha/Computation.h
  include/bits_of_matcha/print.h
  include/bits_of_matcha/Scope.h
  include/bits_of_matcha/rng.h

  include/bits_of_matcha/macros/varargShape.h
  include/bits_of_matcha/macros/generator.h
  include/bits_of_matcha/macros/dataset.h

  include/bits_of_matcha/engine/Tensor.h
  include/bits_of_matcha/engine/Buffer.h
  include/bits_of_matcha/engine/fn.h
  include/bits_of_matcha/engine/Node.h
  include/bits_of_matcha/engine/memory.h
  include/bits_of_matcha/engine/cpu/buffer.h
  include/bits_of_matcha/engine/cpu/memoryPool.h
  include/bits_of_matcha/engine/cpu/blockPool.h
  include/bits_of_matcha/engine/ElementwiseBinary.h
  include/bits_of_matcha/engine/ElementwiseUnary.h
  include/bits_of_matcha/engine/Fold.h
  include/bits_of_matcha/engine/flow/Flow.h
  include/bits_of_matcha/engine/flow/FlowTracer.h
  include/bits_of_matcha/engine/flow/FlowFunctionContext.h

  include/bits_of_matcha/fn/identity.h
  include/bits_of_matcha/fn/basic_arithmetic.h
  include/bits_of_matcha/fn/boolean_arithmetic.h
  include/bits_of_matcha/fn/comparisons.h
  include/bits_of_matcha/fn/dot.h
  include/bits_of_matcha/fn/exponents.h
  include/bits_of_matcha/fn/transpose.h
  include/bits_of_matcha/fn/reshape.h
  include/bits_of_matcha/fn/folds.h
  include/bits_of_matcha/fn/slice.h

  include/bits_of_matcha/data/Instance.h
  include/bits_of_matcha/data/batch.h
  include/bits_of_matcha/data/Dataset.h
  include/bits_of_matcha/dataset/csv.h

  include/bits_of_matcha/nn/Solver.h
  include/bits_of_matcha/nn/Sequential.h
  include/bits_of_matcha/nn/SGD.h
  include/bits_of_matcha/nn/Layers.h

)

set(
  PROJECT_SOURCES

  src/bits_of_matcha/Dtype.cpp
  src/bits_of_matcha/Shape.cpp
  src/bits_of_matcha/Frame.cpp
  src/bits_of_matcha/tensor.cpp
  src/bits_of_matcha/fn.cpp
  src/bits_of_matcha/initializable_fn.cpp
  src/bits_of_matcha/Sequence.cpp
  src/bits_of_matcha/Flow.cpp
  src/bits_of_matcha/Slice.cpp
  src/bits_of_matcha/Device.cpp
  src/bits_of_matcha/print.cpp
  src/bits_of_matcha/Scope.cpp

  src/bits_of_matcha/engine/Tensor.cpp
  src/bits_of_matcha/engine/Buffer.cpp
  src/bits_of_matcha/engine/Node.cpp
  src/bits_of_matcha/engine/fn.cpp
  src/bits_of_matcha/engine/cpu/buffer.cpp
  src/bits_of_matcha/engine/cpu/memoryPool.cpp
  src/bits_of_matcha/engine/cpu/blockPool.cpp
  src/bits_of_matcha/engine/memory.cpp
  src/bits_of_matcha/engine/ElementwiseBinary.cpp
  src/bits_of_matcha/engine/ElementwiseUnary.cpp
  src/bits_of_matcha/engine/Fold.cpp
  src/bits_of_matcha/engine/flow/Flow.cpp
  src/bits_of_matcha/engine/flow/FlowTracer.cpp
  src/bits_of_matcha/engine/flow/FlowFunctionContext.cpp

  src/bits_of_matcha/fn/identity.cpp
  src/bits_of_matcha/fn/basic_arithmetic.cpp
  src/bits_of_matcha/fn/boolean_arithmetic.cpp
  src/bits_of_matcha/fn/comparisons.cpp
  src/bits_of_matcha/fn/dot.cpp
  src/bits_of_matcha/fn/exponents.cpp
  src/bits_of_matcha/fn/transpose.cpp
  src/bits_of_matcha/fn/reshape.cpp
  src/bits_of_matcha/fn/folds.cpp
  src/bits_of_matcha/fn/slice.cpp
  src/bits_of_matcha/rng.cpp

  src/bits_of_matcha/Generator.cpp

  src/bits_of_matcha/data/Instance.cpp
  src/bits_of_matcha/data/batch.cpp
  src/bits_of_matcha/data/Dataset.cpp

  src/bits_of_matcha/dataset/csv.cpp

  src/bits_of_matcha/nn/Sequential.cpp
  src/bits_of_matcha/nn/Layers.cpp
  src/bits_of_matcha/nn/Solver.cpp
  src/bits_of_matcha/nn/SGD.cpp

)


find_package(matcha-device)
find_package(BLAS REQUIRED)
message("${BLAS_LIBRARIES}")


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

add_executable(main src/main.cpp
#  src/todo.cpp
  )

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

target_link_libraries(${PROJECT_NAME} PUBLIC /usr/lib/libcblas.so)
#target_include_directories(${PROJECT_NAME} PRIVATE ${BLAS_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC matcha-device)
#target_link_libraries(${PROJECT_NAME} PUBLIC matcha)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
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
