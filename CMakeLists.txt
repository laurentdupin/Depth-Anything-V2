cmake_minimum_required(VERSION 3.22)

project(depth_anything_v2_c VERSION 0.1.0 LANGUAGES C CXX)
# Keep the selected Vulkan SDK ahead of stale global header installations for
# every runtime, test, and probe target that links the imported Vulkan target.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
include("${CMAKE_CURRENT_LIST_DIR}/../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

if(MSVC)
    # The distributable must be one DLL, so do not require the separately
    # installed Visual C++ runtime DLLs.
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

option(DAV2_BUILD_TESTS "Build the C ABI and preprocessing tests" ON)
option(DAV2_WITH_VULKAN "Build the dependency-free Vulkan executor" ON)
option(DAV2_WITH_METAL "Build the Apple Metal/MPSGraph executor" ${APPLE})
option(DAV2_BUILD_TOOLS "Build native command-line tools" ON)

if(DAV2_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)

    set(DAV2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${DAV2_GENERATED_DIR}")

    function(dav2_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${DAV2_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${DAV2_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_tiled_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_vec4_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=dav2_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM
        )
        set(DAV2_SHADER_HEADERS ${DAV2_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    dav2_compile_shader(elementwise)
    dav2_compile_shader(linear)
    dav2_compile_shader(linear16)
    dav2_compile_shader(linear_half)
    dav2_compile_shader(linear16_half)
    dav2_compile_shader(linear_vec4)
    dav2_compile_shader(linear_vec4_half)
    dav2_compile_shader(linear_vec8)
    dav2_compile_shader(linear_vec8_half)
    dav2_compile_shader(linear_vec8_residual)
    dav2_compile_shader(linear_vec8_half_residual)
    dav2_compile_shader(linear_vec16)
    dav2_compile_shader(linear_vec16_half)
    dav2_compile_shader(linear_vec16_gelu)
    dav2_compile_shader(linear_vec16_residual)
    dav2_compile_shader(linear_vec16_half_residual)
    dav2_compile_shader(pack_fp16)
    dav2_compile_shader(linear_vec16_fp16)
    dav2_compile_shader(linear_vec16_fp16_output_gelu)
    dav2_compile_shader(quantize_rows_int8_fused)
    dav2_compile_shader(linear_int8_tiled)
    dav2_compile_shader(gelu)
    dav2_compile_shader(layer_norm)
    dav2_compile_shader(add_scaled)
    dav2_compile_shader(prepare_tokens)
    dav2_compile_shader(prepare_tokens_fp16)
    dav2_compile_shader(copy_class_token)
    dav2_compile_shader(position_bicubic)
    dav2_compile_shader(add_position)
    dav2_compile_shader(add)
    dav2_compile_shader(bmm)
    dav2_compile_shader(bmm_score_half)
    dav2_compile_shader(bmm_value_half)
    dav2_compile_shader(bmm_score_fp16)
    dav2_compile_shader(bmm_value_fp16)
    dav2_compile_shader(softmax_lastdim)
    dav2_compile_shader(softmax_lastdim_half)
    dav2_compile_shader(project_tokens)
    dav2_compile_shader(project_tokens_half)
    dav2_compile_shader(project_tokens_fp16)
    dav2_compile_shader(conv2d)
    dav2_compile_shader(conv2d8)
    dav2_compile_shader(conv2d_half)
    dav2_compile_shader(conv2d_fp16)
    dav2_compile_shader(conv2d8_half)
    dav2_compile_shader(conv2d_tiled)
    dav2_compile_shader(conv2d_tiled16x8)
    dav2_compile_shader(conv2d_stride2_tiled8x8)
    dav2_compile_shader(conv2d_pointwise_gemm)
    dav2_compile_shader(conv2d8_tiled)
    dav2_compile_shader(conv2d_tiled_half)
    dav2_compile_shader(conv2d8_tiled_half)
    dav2_compile_shader(conv_transpose_nonoverlap)
    dav2_compile_shader(conv_transpose_nonoverlap_half)
    dav2_compile_shader(conv_transpose_nonoverlap_fp16)
    dav2_compile_shader(im2col_quantize_int8)
    dav2_compile_shader(conv_transpose_int8)
    dav2_compile_shader(bilinear_align_true)
    dav2_compile_shader(bilinear_align_true_image)
    dav2_compile_shader(reduce_minmax)
    dav2_compile_shader(normalize_relative)
    dav2_compile_shader(normalize_metric)
    dav2_compile_shader(relu)
    dav2_compile_shader(sigmoid_scale)
    dav2_compile_shader(preprocess_rgba)
    dav2_compile_shader(preprocess_texture)
endif()

add_library(depth_anything_v2 SHARED
    src/dav2.cpp
    src/executor_factory.cpp
    src/inferbridge_harness.cpp
    src/image.cpp
    src/model.cpp
)
add_library(depth_anything_v2::depth_anything_v2 ALIAS depth_anything_v2)
# Vulkan headers must come from the SDK paired with the selected loader. On
# macOS, treating an imported SDK include as a late system directory allows a
# stale /usr/local Vulkan installation to shadow it.
set_property(TARGET depth_anything_v2 PROPERTY NO_SYSTEM_FROM_IMPORTED YES)

target_compile_features(depth_anything_v2 PRIVATE cxx_std_17)
target_include_directories(depth_anything_v2
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(depth_anything_v2
    PRIVATE
        DAV2_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY
        $<$<BOOL:${DAV2_WITH_VULKAN}>:DAV2_WITH_VULKAN=1>
        $<$<BOOL:${DAV2_WITH_METAL}>:DAV2_WITH_METAL=1>
)
set_target_properties(depth_anything_v2 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
if(APPLE)
    # Resolve packaged harness dependencies beside the harness. Never retain
    # the producer machine's absolute addon path.
    set_target_properties(depth_anything_v2 PROPERTIES
        BUILD_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH YES
        INSTALL_RPATH "@loader_path"
    )
endif()

if(DAV2_WITH_VULKAN)
    target_sources(depth_anything_v2 PRIVATE
        src/gpu_model.cpp
        src/encoder.cpp
        src/dpt.cpp
        src/operators.cpp
        src/vulkan.cpp
        src/gpu_preprocess.cpp
        src/vulkan_executor.cpp
        ${DAV2_SHADER_HEADERS}
    )
    target_include_directories(depth_anything_v2 PRIVATE "${DAV2_GENERATED_DIR}")
    target_link_libraries(depth_anything_v2 PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(depth_anything_v2 PRIVATE d3d12 dxgi)
    endif()
endif()

if(DAV2_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "DAV2_WITH_METAL requires an Apple platform")
    endif()
    enable_language(OBJCXX)
    target_sources(depth_anything_v2 PRIVATE src/metal_executor.mm)
    set_source_files_properties(src/metal_executor.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(depth_anything_v2 PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph")
endif()

include(GNUInstallDirs)
install(TARGETS depth_anything_v2
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
    FILES
        include/depth_anything_v2.h
        include/inferbridge_harness.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(DAV2_BUILD_TOOLS)
    add_executable(dav2_run tools/dav2_run.c)
    target_link_libraries(dav2_run PRIVATE depth_anything_v2)
    install(TARGETS dav2_run RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

    if(DAV2_WITH_VULKAN)
        add_executable(dav2_encoder_probe
            tools/dav2_encoder_probe.cpp
            src/encoder.cpp
            src/dpt.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAV2_SHADER_HEADERS}
        )
        target_compile_features(dav2_encoder_probe PRIVATE cxx_std_17)
        target_include_directories(dav2_encoder_probe PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAV2_GENERATED_DIR}"
        )
        target_link_libraries(dav2_encoder_probe PRIVATE Vulkan::Vulkan)

        add_executable(dav2_precision_probe tools/dav2_precision_probe.cpp)
        target_compile_features(dav2_precision_probe PRIVATE cxx_std_17)
        target_link_libraries(dav2_precision_probe PRIVATE depth_anything_v2)
    endif()
endif()

if(DAV2_BUILD_TESTS)
    enable_testing()
    add_executable(dav2_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(dav2_c_abi_smoke PRIVATE depth_anything_v2)
    add_test(NAME dav2_c_abi_smoke COMMAND dav2_c_abi_smoke)

    add_executable(
        dav2_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        dav2_harness_abi_smoke PRIVATE depth_anything_v2)
    target_include_directories(
        dav2_harness_abi_smoke PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(
        NAME dav2_harness_abi_smoke COMMAND dav2_harness_abi_smoke)

    add_executable(
        dav2_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        dav2_harness_full_graph PRIVATE depth_anything_v2)
    target_include_directories(
        dav2_harness_full_graph PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(
        NAME dav2_harness_full_graph COMMAND dav2_harness_full_graph)
    set_tests_properties(
        dav2_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(dav2_image_tests tests/image_tests.cpp src/image.cpp)
    target_include_directories(dav2_image_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME dav2_image_tests COMMAND dav2_image_tests)

    add_executable(dav2_preprocess_probe tools/dav2_preprocess_probe.cpp src/image.cpp)
    target_include_directories(
        dav2_preprocess_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    if(DAV2_WITH_METAL)
        add_executable(dav2_metal_full_graph tests/metal_full_graph.cpp)
        target_link_libraries(dav2_metal_full_graph PRIVATE depth_anything_v2)
        add_test(NAME dav2_metal_full_graph COMMAND dav2_metal_full_graph)
        set_tests_properties(
            dav2_metal_full_graph PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(
            dav2_metal_texture_full_graph
            tests/metal_texture_full_graph.mm)
        target_link_libraries(
            dav2_metal_texture_full_graph PRIVATE depth_anything_v2
            "-framework Foundation"
            "-framework Metal")
        add_test(
            NAME dav2_metal_texture_full_graph
            COMMAND dav2_metal_texture_full_graph)
        set_tests_properties(
            dav2_metal_texture_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if(DAV2_WITH_METAL AND DAV2_WITH_VULKAN)
        add_executable(dav2_backend_parity tests/backend_parity.cpp)
        target_link_libraries(dav2_backend_parity PRIVATE depth_anything_v2)
        add_test(NAME dav2_backend_parity COMMAND dav2_backend_parity)
        set_tests_properties(
            dav2_backend_parity PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if(DAV2_WITH_VULKAN)
        add_executable(dav2_vulkan_smoke
            tests/vulkan_smoke.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAV2_SHADER_HEADERS}
        )
        target_compile_features(dav2_vulkan_smoke PRIVATE cxx_std_17)
        target_include_directories(dav2_vulkan_smoke PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAV2_GENERATED_DIR}"
        )
        target_link_libraries(dav2_vulkan_smoke PRIVATE Vulkan::Vulkan)
        add_test(NAME dav2_vulkan_smoke COMMAND dav2_vulkan_smoke)

        if(WIN32)
            add_executable(
                dav2_model_metadata_tests
                tests/model_metadata.cpp
                src/model.cpp)
            target_compile_features(
                dav2_model_metadata_tests PRIVATE cxx_std_17)
            target_include_directories(
                dav2_model_metadata_tests PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src")
            add_test(
                NAME dav2_model_metadata_tests
                COMMAND dav2_model_metadata_tests)

            add_executable(dav2_vulkan_external_smoke
                tests/vulkan_external_smoke.cpp
                src/gpu_preprocess.cpp
                src/image.cpp
                src/vulkan.cpp
                ${DAV2_SHADER_HEADERS}
            )
            target_compile_features(
                dav2_vulkan_external_smoke PRIVATE cxx_std_17)
            target_include_directories(
                dav2_vulkan_external_smoke PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${DAV2_GENERATED_DIR}"
            )
            target_link_libraries(
                dav2_vulkan_external_smoke PRIVATE
                    Vulkan::Vulkan d3d12 dxgi)
            add_test(
                NAME dav2_vulkan_external_smoke
                COMMAND dav2_vulkan_external_smoke)
            set_tests_properties(
                dav2_vulkan_external_smoke
                PROPERTIES SKIP_RETURN_CODE 77)

            add_executable(
                dav2_d3d12_full_graph
                tests/d3d12_full_graph.cpp)
            target_compile_features(
                dav2_d3d12_full_graph PRIVATE cxx_std_17)
            target_link_libraries(
                dav2_d3d12_full_graph PRIVATE
                    depth_anything_v2 d3d12 dxgi)
            target_compile_definitions(
                dav2_d3d12_full_graph PRIVATE
                    "DAV2_DEFAULT_VITS_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/validation/depth_anything_v2_vits.dav2\"")
            add_test(
                NAME dav2_d3d12_full_graph
                COMMAND dav2_d3d12_full_graph)
            set_tests_properties(
                dav2_d3d12_full_graph
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
