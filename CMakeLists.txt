cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(score_addon_librediffusion LANGUAGES CXX)

# Addon requires CUDA Toolkit; only Windows / Linux.
if((NOT WIN32) AND (NOT LINUX))
  return()
endif()

# Use the Avendish the host (ossia score) provides; otherwise fetch it.
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    GIT_TAG main)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

if(COMMAND score_common_setup)
  score_common_setup()
endif()

add_library(librediffusion_deps INTERFACE)
target_include_directories(librediffusion_deps INTERFACE 3rdparty/librediffusion/src)

avnd_addon_init(NAME score_addon_librediffusion)

avnd_addon_object(
  BASE score_addon_librediffusion C_NAME streamdiffusion CLASS StreamDiffusion NAMESPACE lo
  DEPENDENCIES librediffusion_deps
  SOURCES
    LibreDiffusion/LibreDiffusion.hpp
    LibreDiffusion/LibreDiffusion.cpp
    LibreDiffusion/EngineCache.hpp
    LibreDiffusion/EngineCache.cpp)

avnd_addon_finalize(NAME score_addon_librediffusion UUID b3fe4d80-0a1a-4b23-8cd2-b82472f9f91f VERSION 1)
