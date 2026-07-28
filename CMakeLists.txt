cmake_minimum_required(VERSION 3.25)

project(depth_anything_v2_c VERSION 0.1.0 LANGUAGES C CXX)

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
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
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
    dav2_compile_shader(gelu)
    dav2_compile_shader(layer_norm)
    dav2_compile_shader(add_scaled)
    dav2_compile_shader(prepare_tokens)
    dav2_compile_shader(position_bicubic)
    dav2_compile_shader(add_position)
    dav2_compile_shader(add)
    dav2_compile_shader(bmm)
    dav2_compile_shader(bmm_score_half)
    dav2_compile_shader(bmm_value_half)
    dav2_compile_shader(softmax_lastdim)
    dav2_compile_shader(softmax_lastdim_half)
    dav2_compile_shader(project_tokens)
    dav2_compile_shader(project_tokens_half)
    dav2_compile_shader(conv2d)
    dav2_compile_shader(conv2d8)
    dav2_compile_shader(conv2d_half)
    dav2_compile_shader(conv2d8_half)
    dav2_compile_shader(conv_transpose_nonoverlap)
    dav2_compile_shader(conv_transpose_nonoverlap_half)
    dav2_compile_shader(bilinear_align_true)
    dav2_compile_shader(relu)
endif()

add_library(depth_anything_v2 SHARED
    src/dav2.cpp
    src/image.cpp
)
add_library(depth_anything_v2::depth_anything_v2 ALIAS depth_anything_v2)

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
        $<$<BOOL:${DAV2_WITH_VULKAN}>:DAV2_WITH_VULKAN=1>
)
set_target_properties(depth_anything_v2 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(DAV2_WITH_VULKAN)
    target_sources(depth_anything_v2 PRIVATE
        src/model.cpp
        src/gpu_model.cpp
        src/encoder.cpp
        src/dpt.cpp
        src/operators.cpp
        src/vulkan.cpp
        src/vulkan_executor.cpp
        ${DAV2_SHADER_HEADERS}
    )
    target_include_directories(depth_anything_v2 PRIVATE "${DAV2_GENERATED_DIR}")
    target_link_libraries(depth_anything_v2 PRIVATE Vulkan::Vulkan)
endif()

include(GNUInstallDirs)
install(TARGETS depth_anything_v2
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES include/depth_anything_v2.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

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
            ${DAV2_SHADER_HEADERS}
        )
        target_compile_features(dav2_encoder_probe PRIVATE cxx_std_17)
        target_include_directories(dav2_encoder_probe PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAV2_GENERATED_DIR}"
        )
        target_link_libraries(dav2_encoder_probe PRIVATE Vulkan::Vulkan)
    endif()
endif()

if(DAV2_BUILD_TESTS)
    enable_testing()
    add_executable(dav2_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(dav2_c_abi_smoke PRIVATE depth_anything_v2)
    add_test(NAME dav2_c_abi_smoke COMMAND dav2_c_abi_smoke)

    add_executable(dav2_image_tests tests/image_tests.cpp src/image.cpp)
    target_include_directories(dav2_image_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME dav2_image_tests COMMAND dav2_image_tests)

    add_executable(dav2_preprocess_probe tools/dav2_preprocess_probe.cpp src/image.cpp)
    target_include_directories(
        dav2_preprocess_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    if(DAV2_WITH_VULKAN)
        add_executable(dav2_vulkan_smoke
            tests/vulkan_smoke.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
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
    endif()
endif()
