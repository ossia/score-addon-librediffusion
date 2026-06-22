cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(score_addon_librediffusion LANGUAGES CXX)

# Max/MSP externals link the static CRT (/MT) on Windows, as the official
# max-sdk-base requires. Set it for the whole addon -- before Avendish and the
# object library are created -- so every target shares one runtime; mixing /MT
# and /MD trips MSVC with LNK2038. (CMP0091 is NEW via cmake_minimum_required.)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Addon requires CUDA Toolkit; only Windows / Linux. On other platforms (macOS)
# the object cannot be built at all. Register a no-op install rule before bailing
# out so the score-addon CI's `cmake --build --target install` step is a clean
# success instead of failing with "No rule to make target install".
if((NOT WIN32) AND (NOT LINUX))
  install(CODE "message(STATUS \"score-addon-librediffusion: skipped on this platform (requires CUDA)\")")
  return()
endif()

# Use the Avendish the host (ossia score) provides; otherwise fetch it.
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    GIT_TAG ebbbdf81a3c69831c50231877814b63852f9881c)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

if(COMMAND score_common_setup)
  score_common_setup()
endif()

# The StreamDiffusion object is deeply coupled to the ossia score stack: it parses
# controls through score's State::parseValue, and pulls libossia headers that
# only the score build (or a
# full libossia checkout) provides. The pure-Avendish standalone back-ends
# (pd / max / python / touchdesigner / godot) ship none of those, so the object
# cannot be built there. Register it only when building against score; standalone
# configures to a clean no-op (the addon then ships nothing, as intended).
if(TARGET score_plugin_avnd OR COMMAND avnd_score_plugin_add)
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
endif()
