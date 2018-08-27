set(TAICHI_PROJECT_NAME "differentiable_mpm")

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h" "src/*/*.h" "src/*/*.cpp")
file(GLOB_RECURSE PROJECT_SOURCES_CUDA "src/*.cu")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

if ($ENV{CUDA_ARCH})
    message("CUDA Arch: $ENV{CUDA_ARCH}")
    find_package(CUDA 8.0 REQUIRED)
    set (CUDA_HOST_COMPILER g++)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=compute_$ENV{CUDA_ARCH} -code=sm_$ENV{CUDA_ARCH} -Xcompiler "-fPIC" --maxrregcount 64 --use_fast_math --ptxas-options=-allow-expensive-optimizations=true,-O3)
    if ($ENV{CUDA_ARCH} LESS 60)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
    else()
        # __CUDA_ARCH__ is 0 here, and cannot be redefined...
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D__CUDA_ARCH___=$ENV{CUDA_ARCH}0)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    cuda_add_library(cudmpm ${PROJECT_SOURCES_CUDA})
else()
    message("Environment variable CUDA_ARCH not found. Not using CUDA.")
endif()

add_library(taichi_${TAICHI_PROJECT_NAME} SHARED ${PROJECT_SOURCES})

target_link_libraries(taichi_${TAICHI_PROJECT_NAME} ${CORE_LIBRARY_NAME} ${SHARED_LIBS} cudmpm)

if ($ENV{CUDA_ARCH})
    target_link_libraries(taichi_${TAICHI_PROJECT_NAME})
endif()

if ($ENV{TC_NEED_ZLIB})
    target_link_libraries(taichi_${TAICHI_PROJECT_NAME} z)
endif()

