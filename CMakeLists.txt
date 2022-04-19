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

  include/bits_of_matcha/Dtype.h
  include/bits_of_matcha/Shape.h
  include/bits_of_matcha/Frame.h
  include/bits_of_matcha/tensor.h
  include/bits_of_matcha/ops.h
  include/bits_of_matcha/Device.h
  include/bits_of_matcha/print.h
  include/bits_of_matcha/Flow.h

  include/bits_of_matcha/engine/memory/memory.h
  include/bits_of_matcha/engine/memory/Buffer.h

  include/bits_of_matcha/engine/tensor/Tensor.h
  include/bits_of_matcha/engine/tensor/TensorCtx.h
  include/bits_of_matcha/engine/tensor/RefReqCounted.h
  include/bits_of_matcha/engine/tensor/iterations.h

  include/bits_of_matcha/engine/op/Op.h
  include/bits_of_matcha/engine/op/OpInputs.h
  include/bits_of_matcha/engine/op/OpOutputs.h
  include/bits_of_matcha/engine/op/OpMeta.h
  include/bits_of_matcha/engine/op/Ops.h

  include/bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h
  include/bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h

  include/bits_of_matcha/engine/flow/Flow.h
  include/bits_of_matcha/engine/flow/Tracer.h
  include/bits_of_matcha/engine/flow/Tasks.h
  include/bits_of_matcha/engine/flow/graph/Graph.h
  include/bits_of_matcha/engine/flow/graph/OpDict.h
  include/bits_of_matcha/engine/flow/graph/OpMask.h
  include/bits_of_matcha/engine/flow/graph/TensorDict.h
  include/bits_of_matcha/engine/flow/graph/TensorMask.h
  include/bits_of_matcha/engine/flow/compiler/Compiler.h

  include/bits_of_matcha/engine/utils/TexGraph.h

  include/bits_of_matcha/engine/cpu/Buffer.h
  include/bits_of_matcha/engine/cpu/MemoryPool.h
  include/bits_of_matcha/engine/cpu/BlockPool.h

  include/bits_of_matcha/engine/ops/Add.h
  include/bits_of_matcha/engine/ops/AddBack.h
  include/bits_of_matcha/engine/ops/Dot.h
  include/bits_of_matcha/engine/ops/DotBack.h
  include/bits_of_matcha/engine/ops/Transpose.h
  include/bits_of_matcha/engine/ops/TransposeBack.h
)

set(
  PROJECT_SOURCES

  src/bits_of_matcha/Dtype.cpp
  src/bits_of_matcha/Shape.cpp
  src/bits_of_matcha/Frame.cpp
  src/bits_of_matcha/tensor.cpp
  src/bits_of_matcha/ops.cpp
  src/bits_of_matcha/Device.cpp
  src/bits_of_matcha/Flow.cpp

  src/bits_of_matcha/engine/memory/memory.cpp
  src/bits_of_matcha/engine/memory/Buffer.cpp

  src/bits_of_matcha/engine/tensor/Tensor.cpp
  src/bits_of_matcha/engine/tensor/TensorCtx.cpp
  src/bits_of_matcha/engine/tensor/RefReqCounted.cpp
  src/bits_of_matcha/engine/tensor/iterations.cpp

  src/bits_of_matcha/engine/op/Op.cpp
  src/bits_of_matcha/engine/autograd/OpBack.cpp
  src/bits_of_matcha/engine/autograd/AccumulateGrads.cpp

  src/bits_of_matcha/engine/op/OpCtx.cpp
  src/bits_of_matcha/engine/op/OpInputs.cpp
  src/bits_of_matcha/engine/op/OpOutputs.cpp
  src/bits_of_matcha/engine/op/Ops.cpp

  src/bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.cpp
  src/bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.cpp

  src/bits_of_matcha/engine/flow/Flow.cpp
  src/bits_of_matcha/engine/flow/Tracer.cpp
  src/bits_of_matcha/engine/flow/Tasks.cpp
  src/bits_of_matcha/engine/flow/graph/Graph.cpp
  src/bits_of_matcha/engine/flow/graph/OpDict.cpp
  src/bits_of_matcha/engine/flow/graph/OpMask.cpp
  src/bits_of_matcha/engine/flow/graph/TensorDict.cpp
  src/bits_of_matcha/engine/flow/graph/TensorMask.cpp
  src/bits_of_matcha/engine/flow/compiler/Compiler.cpp

  src/bits_of_matcha/engine/utils/TexGraph.cpp

  src/bits_of_matcha/engine/cpu/Buffer.cpp
  src/bits_of_matcha/engine/cpu/MemoryPool.cpp
  src/bits_of_matcha/engine/cpu/BlockPool.cpp

  src/bits_of_matcha/engine/ops/Add.cpp
  src/bits_of_matcha/engine/ops/AddBack.cpp
  src/bits_of_matcha/engine/ops/Dot.cpp
  src/bits_of_matcha/engine/ops/DotBack.cpp
  src/bits_of_matcha/engine/ops/Transpose.cpp
  src/bits_of_matcha/engine/ops/TransposeBack.cpp
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
