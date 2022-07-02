cmake_minimum_required(VERSION 3.17)
project(
  matcha VERSION 0.0.0
  DESCRIPTION "Attachable Machine Learning."
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pg)
set(CMAKE_CXX_FLAGS -march=native)
set(CMAKE_CXX_FLAGS -O2)

set(
  PROJECT_HEADERS

  include/matcha/matcha
  include/matcha/tensor
  include/matcha/dataset
  include/matcha/nn

  include/bits_of_matcha/Dtype.h
  include/bits_of_matcha/Shape.h
  include/bits_of_matcha/Frame.h
  include/bits_of_matcha/tensor.h
  include/bits_of_matcha/ops.h
  include/bits_of_matcha/Device.h
  include/bits_of_matcha/print.h
  include/bits_of_matcha/random.h
  include/bits_of_matcha/Flow.h
  include/bits_of_matcha/Backprop.h
  include/bits_of_matcha/Loader.h
  include/bits_of_matcha/Engine.h

  include/bits_of_matcha/error/Error.h
  include/bits_of_matcha/error/IncompatibleDtypesError.h
  include/bits_of_matcha/error/IncompatibleShapesError.h
  include/bits_of_matcha/error/BroadcastError.h
  include/bits_of_matcha/error/ImplicitCastError.h

  include/bits_of_matcha/macros/generator.h
  include/bits_of_matcha/macros/dataset.h
  include/bits_of_matcha/macros/vararg_shape.h
  include/bits_of_matcha/macros/vararg_tensors.h

  include/bits_of_matcha/dataset/Instance.h
  include/bits_of_matcha/dataset/InstanceIterator.h
  include/bits_of_matcha/dataset/Dataset.h
  include/bits_of_matcha/dataset/loaders/Tensors.h
  include/bits_of_matcha/dataset/loaders/Generator.h
  include/bits_of_matcha/dataset/loaders/Take.h
  include/bits_of_matcha/dataset/loaders/Batch.h
  include/bits_of_matcha/dataset/loaders/Map.h
  include/bits_of_matcha/dataset/loaders/Csv.h

  include/bits_of_matcha/nn/Net.h

  include/bits_of_matcha/nn/initializers.h
  include/bits_of_matcha/nn/activations.h

  include/bits_of_matcha/nn/Layer.h
  include/bits_of_matcha/nn/layers/Flatten.h
  include/bits_of_matcha/nn/layers/Linear.h
  include/bits_of_matcha/nn/layers/Fc.h
  include/bits_of_matcha/nn/layers/BatchNorm.h

  include/bits_of_matcha/nn/Optimizer.h
  include/bits_of_matcha/nn/optimizers/Sgd.h

  include/bits_of_matcha/nn/Loss.h
  include/bits_of_matcha/nn/losses/CategoricalCrossentropy.h

  include/bits_of_matcha/nn/Callback.h
  include/bits_of_matcha/nn/callbacks/Logger.h

  include/bits_of_matcha/engine/memory/memory.h
  include/bits_of_matcha/engine/memory/Block.h
  include/bits_of_matcha/engine/memory/Buffer.h
  include/bits_of_matcha/engine/memory/cast.h

  include/bits_of_matcha/engine/tensor/Tensor.h
  include/bits_of_matcha/engine/tensor/factories.h
  include/bits_of_matcha/engine/tensor/cast.h
  include/bits_of_matcha/engine/tensor/iterations.h
  include/bits_of_matcha/engine/tensor/RefReqCounted.h

  include/bits_of_matcha/engine/op/Op.h
  include/bits_of_matcha/engine/op/OpInputs.h
  include/bits_of_matcha/engine/op/OpOutputs.h
  include/bits_of_matcha/engine/op/OpMeta.h
  include/bits_of_matcha/engine/op/Ops.h
  include/bits_of_matcha/engine/op/BackCtx.h
  include/bits_of_matcha/engine/op/OpBack.h

  include/bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h
  include/bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h
  include/bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.h
  include/bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h

  include/bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h
  include/bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h
  include/bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h
  include/bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h

  include/bits_of_matcha/engine/flow/Flow.h
  include/bits_of_matcha/engine/flow/Graph.h
  include/bits_of_matcha/engine/flow/Tracer.h
  include/bits_of_matcha/engine/flow/module/Module.h
  include/bits_of_matcha/engine/flow/module/ModuleForw.h
  include/bits_of_matcha/engine/flow/module/ModuleBack.h
  include/bits_of_matcha/engine/flow/Tasks.h
  include/bits_of_matcha/engine/flow/Backprop.h

  include/bits_of_matcha/engine/autograd/gradient.h
  include/bits_of_matcha/engine/autograd/Autograd.h
  include/bits_of_matcha/engine/autograd/AccumulateGrads.h

#  include/bits_of_matcha/engine/utils/TexGraph.h

  include/bits_of_matcha/engine/cpu/Buffer.h
  include/bits_of_matcha/engine/cpu/MemoryPool.h
  include/bits_of_matcha/engine/cpu/BlockPool.h

  include/bits_of_matcha/engine/cpu/kernels/elementwiseUnary.h
  include/bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h
  include/bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h
  include/bits_of_matcha/engine/cpu/kernels/fill.h
  include/bits_of_matcha/engine/cpu/kernels/transpose.h
  include/bits_of_matcha/engine/cpu/kernels/mm.h

  include/bits_of_matcha/engine/dataset/Dataset.h

  include/bits_of_matcha/engine/ops/Add.h
  include/bits_of_matcha/engine/ops/Multiply.h
  include/bits_of_matcha/engine/ops/Divide.h
  include/bits_of_matcha/engine/ops/Dot.h
  include/bits_of_matcha/engine/ops/Transpose.h
  include/bits_of_matcha/engine/ops/Identity.h
  include/bits_of_matcha/engine/ops/Reshape.h
  include/bits_of_matcha/engine/ops/Print.h
  include/bits_of_matcha/engine/ops/Pow.h
  include/bits_of_matcha/engine/ops/Exp.h
  include/bits_of_matcha/engine/ops/SaveImage.h
  include/bits_of_matcha/engine/ops/LoadImage.h
  include/bits_of_matcha/engine/ops/SaveCsv.h
  include/bits_of_matcha/engine/ops/LoadCsv.h
  include/bits_of_matcha/engine/ops/Max.h
  include/bits_of_matcha/engine/ops/MaxBetween.h
  include/bits_of_matcha/engine/ops/Min.h
  include/bits_of_matcha/engine/ops/MinBetween.h
  include/bits_of_matcha/engine/ops/Eq.h
  include/bits_of_matcha/engine/ops/Neq.h
  include/bits_of_matcha/engine/ops/Lt.h
  include/bits_of_matcha/engine/ops/Le.h
  include/bits_of_matcha/engine/ops/Gt.h
  include/bits_of_matcha/engine/ops/Ge.h
  include/bits_of_matcha/engine/ops/Argmax.h
  include/bits_of_matcha/engine/ops/Argmin.h
  include/bits_of_matcha/engine/ops/Sum.h
  include/bits_of_matcha/engine/ops/Product.h
  include/bits_of_matcha/engine/ops/Uniform.h
  include/bits_of_matcha/engine/ops/Normal.h
  include/bits_of_matcha/engine/ops/Stack.h
  include/bits_of_matcha/engine/ops/Cast.h

  include/bits_of_matcha/grafana/Grafana.h
)

set(
  PROJECT_SOURCES

  src/bits_of_matcha/Dtype.cpp
  src/bits_of_matcha/Shape.cpp
  src/bits_of_matcha/Frame.cpp
  src/bits_of_matcha/tensor.cpp
  src/bits_of_matcha/ops.cpp
  src/bits_of_matcha/Device.cpp
  src/bits_of_matcha/random.cpp
  src/bits_of_matcha/Flow.cpp
  src/bits_of_matcha/Backprop.cpp
  src/bits_of_matcha/Loader.cpp
  src/bits_of_matcha/Engine.cpp

  src/bits_of_matcha/error/Error.cpp
  src/bits_of_matcha/error/IncompatibleDtypesError.cpp
  src/bits_of_matcha/error/IncompatibleShapesError.cpp
  src/bits_of_matcha/error/BroadcastError.cpp
  src/bits_of_matcha/error/ImplicitCastError.cpp

  src/bits_of_matcha/dataset/Instance.cpp
  src/bits_of_matcha/dataset/InstanceIterator.cpp
  src/bits_of_matcha/dataset/Dataset.cpp
  src/bits_of_matcha/dataset/loaders/Tensors.cpp
  src/bits_of_matcha/dataset/loaders/Generator.cpp
  src/bits_of_matcha/dataset/loaders/Take.cpp
  src/bits_of_matcha/dataset/loaders/Batch.cpp
  src/bits_of_matcha/dataset/loaders/Map.cpp
  src/bits_of_matcha/dataset/loaders/Csv.cpp

  src/bits_of_matcha/nn/Net.cpp
  src/bits_of_matcha/nn/activations.cpp
  src/bits_of_matcha/nn/Layer.cpp
  src/bits_of_matcha/nn/layers/Flatten.cpp
  src/bits_of_matcha/nn/layers/Linear.cpp
  src/bits_of_matcha/nn/layers/Fc.cpp
  src/bits_of_matcha/nn/layers/BatchNorm.cpp
  src/bits_of_matcha/nn/optimizers/Sgd.cpp
  src/bits_of_matcha/nn/callbacks/Logger.cpp

  src/bits_of_matcha/engine/memory/memory.cpp
  src/bits_of_matcha/engine/memory/Block.cpp
  src/bits_of_matcha/engine/memory/Buffer.cpp

  src/bits_of_matcha/engine/tensor/Tensor.cpp
  src/bits_of_matcha/engine/tensor/factories.cpp
  src/bits_of_matcha/engine/tensor/cast.cpp
  src/bits_of_matcha/engine/tensor/iterations.cpp
  src/bits_of_matcha/engine/tensor/RefReqCounted.cpp

  src/bits_of_matcha/engine/op/Op.cpp
  src/bits_of_matcha/engine/op/OpBack.cpp
  src/bits_of_matcha/engine/op/OpInputs.cpp
  src/bits_of_matcha/engine/op/OpOutputs.cpp
  src/bits_of_matcha/engine/op/Ops.cpp

  src/bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.cpp
  src/bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.cpp
  src/bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.cpp
  src/bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.cpp
  src/bits_of_matcha/engine/iterations/AxiswiseFoldCtx.cpp

  src/bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.cpp
  src/bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.cpp

  src/bits_of_matcha/engine/flow/Flow.cpp
  src/bits_of_matcha/engine/flow/Graph.cpp
  src/bits_of_matcha/engine/flow/Tasks.cpp
  src/bits_of_matcha/engine/flow/Tracer.cpp
  src/bits_of_matcha/engine/flow/module/Module.cpp
  src/bits_of_matcha/engine/flow/module/ModuleForw.cpp
  src/bits_of_matcha/engine/flow/module/ModuleBack.cpp

  src/bits_of_matcha/engine/autograd/gradient.cpp
  src/bits_of_matcha/engine/autograd/Autograd.cpp
  src/bits_of_matcha/engine/autograd/AccumulateGrads.cpp
  src/bits_of_matcha/engine/autograd/Autograd.cpp

#  src/bits_of_matcha/engine/utils/TexGraph.cpp

  src/bits_of_matcha/engine/cpu/Block.cpp
  src/bits_of_matcha/engine/cpu/MemoryPool.cpp
  src/bits_of_matcha/engine/cpu/BlockPool.cpp

  src/bits_of_matcha/engine/dataset/Dataset.cpp

  src/bits_of_matcha/engine/ops/Add.cpp
  src/bits_of_matcha/engine/ops/Multiply.cpp
  src/bits_of_matcha/engine/ops/Divide.cpp
  src/bits_of_matcha/engine/ops/Dot.cpp
  src/bits_of_matcha/engine/ops/Transpose.cpp
  src/bits_of_matcha/engine/ops/Identity.cpp
  src/bits_of_matcha/engine/ops/Reshape.cpp
  src/bits_of_matcha/engine/ops/Print.cpp
  src/bits_of_matcha/engine/ops/Pow.cpp
  src/bits_of_matcha/engine/ops/Exp.cpp
  src/bits_of_matcha/engine/ops/LoadImage.cpp
  src/bits_of_matcha/engine/ops/SaveImage.cpp
  src/bits_of_matcha/engine/ops/LoadCsv.cpp
  src/bits_of_matcha/engine/ops/SaveCsv.cpp
  src/bits_of_matcha/engine/ops/Max.cpp
  src/bits_of_matcha/engine/ops/MaxBetween.cpp
  src/bits_of_matcha/engine/ops/Min.cpp
  src/bits_of_matcha/engine/ops/MinBetween.cpp
  src/bits_of_matcha/engine/ops/Eq.cpp
  src/bits_of_matcha/engine/ops/Neq.cpp
  src/bits_of_matcha/engine/ops/Lt.cpp
  src/bits_of_matcha/engine/ops/Le.cpp
  src/bits_of_matcha/engine/ops/Gt.cpp
  src/bits_of_matcha/engine/ops/Ge.cpp
  src/bits_of_matcha/engine/ops/Argmax.cpp
  src/bits_of_matcha/engine/ops/Argmin.cpp
  src/bits_of_matcha/engine/ops/Sum.cpp
  src/bits_of_matcha/engine/ops/Product.cpp
  src/bits_of_matcha/engine/ops/Uniform.cpp
  src/bits_of_matcha/engine/ops/Normal.cpp
  src/bits_of_matcha/engine/ops/Stack.cpp
  src/bits_of_matcha/engine/ops/Cast.cpp

  src/bits_of_matcha/grafana/Grafana.cpp
)

#set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(OpenCL REQUIRED)
find_package(libpng CONFIG REQUIRED)
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

add_executable(main src/main.cpp)

target_include_directories(
  main
  PRIVATE
    include/
)

target_link_libraries(${PROJECT_NAME} PUBLIC png)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
#target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/lib/libcblas.so)
target_link_libraries(${PROJECT_NAME} PRIVATE dl)
target_link_libraries(${PROJECT_NAME} PRIVATE curl)

target_link_libraries(main PUBLIC ${PROJECT_NAME})


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/MatchaConfigVersion.cmake"
  VERSION 0.1
  COMPATIBILITY AnyNewerVersion
)

install(TARGETS ${PROJECT_NAME}
  EXPORT MatchaTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION include COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/MatchaConfig.cmake"
  "${PROJECT_BINARY_DIR}/MatchaConfig.cmake"
  INSTALL_DESTINATION lib/cmake/matcha
)

install(EXPORT MatchaTargets DESTINATION lib/cmake/matcha)
install(FILES "${PROJECT_BINARY_DIR}/MatchaConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/MatchaConfig.cmake"
  DESTINATION lib/cmake/matcha-engine
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
