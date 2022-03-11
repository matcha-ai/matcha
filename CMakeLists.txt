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

  include/bits_of_matcha/error/error.h

  include/bits_of_matcha/dtype.h
  include/bits_of_matcha/shape.h
  include/bits_of_matcha/frame.h
  include/bits_of_matcha/tensor.h
  include/bits_of_matcha/params.h
  include/bits_of_matcha/fn.h
  include/bits_of_matcha/flow.h
  include/bits_of_matcha/flowHost.h
  include/bits_of_matcha/slice.h
  include/bits_of_matcha/stream.h
  include/bits_of_matcha/random.h
  include/bits_of_matcha/dataset.h
  include/bits_of_matcha/device.h
  include/bits_of_matcha/computation.h
  include/bits_of_matcha/print.h
  include/bits_of_matcha/scope.h

  include/bits_of_matcha/varargShape.h

  include/bits_of_matcha/engine/tensor.h
  include/bits_of_matcha/engine/buffer.h
  include/bits_of_matcha/engine/fn.h
  include/bits_of_matcha/engine/node.h
  include/bits_of_matcha/engine/flow.h
  include/bits_of_matcha/engine/flowContext.h
  include/bits_of_matcha/engine/flowBuilder.h
  include/bits_of_matcha/engine/memory.h
  include/bits_of_matcha/engine/cpu/buffer.h
  include/bits_of_matcha/engine/cpu/memoryPool.h
  include/bits_of_matcha/engine/cpu/blockPool.h
  include/bits_of_matcha/engine/elementwiseBinary.h
  include/bits_of_matcha/engine/elementwiseUnary.h
  include/bits_of_matcha/engine/fold.h

  include/bits_of_matcha/fn/identity.h
  include/bits_of_matcha/fn/basicArithmetic.h
  include/bits_of_matcha/fn/booleanArithmetic.h
  include/bits_of_matcha/fn/comparisons.h
  include/bits_of_matcha/fn/dot.h
  include/bits_of_matcha/fn/exponents.h
  include/bits_of_matcha/fn/transpose.h
  include/bits_of_matcha/fn/reshape.h
  include/bits_of_matcha/fn/folds.h

)

set(
  PROJECT_SOURCES

  src/bits_of_matcha/dtype.cpp
  src/bits_of_matcha/shape.cpp
  src/bits_of_matcha/frame.cpp
  src/bits_of_matcha/tensor.cpp
  src/bits_of_matcha/fn.cpp
  src/bits_of_matcha/flow.cpp
  src/bits_of_matcha/slice.cpp
  src/bits_of_matcha/stream.cpp
  src/bits_of_matcha/random.cpp
  src/bits_of_matcha/dataset.cpp
  src/bits_of_matcha/device.cpp
  src/bits_of_matcha/print.cpp
  src/bits_of_matcha/scope.cpp

  src/bits_of_matcha/engine/tensor.cpp
  src/bits_of_matcha/engine/buffer.cpp
  src/bits_of_matcha/engine/node.cpp
  src/bits_of_matcha/engine/fn.cpp
  src/bits_of_matcha/engine/flow.cpp
  src/bits_of_matcha/engine/flowContext.cpp
  src/bits_of_matcha/engine/flowBuilder.cpp
  src/bits_of_matcha/engine/pipeBuffers.cpp
  src/bits_of_matcha/engine/cpu/buffer.cpp
  src/bits_of_matcha/engine/cpu/memoryPool.cpp
  src/bits_of_matcha/engine/cpu/blockPool.cpp
  src/bits_of_matcha/engine/memory.cpp
  src/bits_of_matcha/engine/elementwiseBinary.cpp
  src/bits_of_matcha/engine/elementwiseUnary.cpp
  src/bits_of_matcha/engine/fold.cpp

  src/bits_of_matcha/fn/identity.cpp
  src/bits_of_matcha/fn/basicArithmetic.cpp
  src/bits_of_matcha/fn/booleanArithmetic.cpp
  src/bits_of_matcha/fn/comparisons.cpp
  src/bits_of_matcha/fn/dot.cpp
  src/bits_of_matcha/fn/exponents.cpp
  src/bits_of_matcha/fn/transpose.cpp
  src/bits_of_matcha/fn/reshape.cpp
  src/bits_of_matcha/fn/folds.cpp

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
