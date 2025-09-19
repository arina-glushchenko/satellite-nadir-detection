cmake_minimum_required(VERSION 3.10)
project(rknn_demo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKNN API paths
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/include/lib/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

# OpenCV configuration
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/include/lib/opencv-mobile-4.10.0-luckfox-pico/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_PATH}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lib
    ${CMAKE_SOURCE_DIR}/include/stb
)

# Source files
set(SOURCES
    src/main.cc
    src/postprocess.cc
    src/draw_utils_stb.cc
    src/stb_impl.cc
)

# Create executable
add_executable(rknn_demo ${SOURCES})

# Link libraries
target_link_libraries(rknn_demo
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    m
)

# Installation
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_demo_${CMAKE_SYSTEM_NAME})
set(CMAKE_INSTALL_RPATH "lib")

install(TARGETS rknn_demo DESTINATION .)
install(DIRECTORY model/${TARGET_SOC} DESTINATION model)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)

# Install images (if any)
file(GLOB IMAGE_FILES "model/*.jpg")
if(IMAGE_FILES)
    install(FILES ${IMAGE_FILES} DESTINATION model)
endif()