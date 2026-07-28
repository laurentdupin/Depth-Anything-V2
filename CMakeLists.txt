cmake_minimum_required(VERSION 3.25)

project(depth_anything_v2_c VERSION 0.1.0 LANGUAGES C CXX)

option(DAV2_BUILD_TESTS "Build the C ABI and preprocessing tests" ON)
option(DAV2_WITH_TORCH "Build the LibTorch/Vulkan model executor" OFF)

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
        $<$<BOOL:${DAV2_WITH_TORCH}>:DAV2_WITH_TORCH=1>
)
set_target_properties(depth_anything_v2 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(DAV2_WITH_TORCH)
    find_package(Torch CONFIG REQUIRED)
    target_sources(depth_anything_v2 PRIVATE src/torch_executor.cpp)
    target_link_libraries(depth_anything_v2 PRIVATE "${TORCH_LIBRARIES}")
endif()

include(GNUInstallDirs)
install(TARGETS depth_anything_v2
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES include/depth_anything_v2.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(DAV2_BUILD_TESTS)
    enable_testing()
    add_executable(dav2_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(dav2_c_abi_smoke PRIVATE depth_anything_v2)
    add_test(NAME dav2_c_abi_smoke COMMAND dav2_c_abi_smoke)

    add_executable(dav2_image_tests tests/image_tests.cpp src/image.cpp)
    target_include_directories(dav2_image_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME dav2_image_tests COMMAND dav2_image_tests)
endif()
