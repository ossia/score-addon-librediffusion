cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(score_addon_librediffusion LANGUAGES CXX)

# Max/MSP externals link the static CRT (/MT) on Windows, as the official
# max-sdk-base requires. Set it for the whole addon -- before Avendish and the
# object library are created -- so every target shares one runtime; mixing /MT
# and /MD trips MSVC with LNK2038. (CMP0091 is NEW via cmake_minimum_required.)
#
# Only for the standalone build: inside a score build the Max backend is not
# emitted, and score links the dynamic CRT (/MD). Forcing /MT here would make
# score_addon_librediffusion.lib mismatch the rest of score (LNK2038:
# MT_StaticRelease vs MD_DynamicRelease), so defer to score's runtime instead.
if(MSVC AND NOT TARGET score_lib_base)
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
    GIT_TAG 480759886e295127c6bb2e6f62ba6af0c84ac145)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

if(COMMAND score_common_setup)
  score_common_setup()
endif()

# The StreamDiffusion object loads the librediffusion runtime through its C API
# by dynamic loading, so it needs no CUDA/TensorRT at build time -- only the C
# API header below. The sources have no hard score dependency: the one ossia
# include (ossia/detail/fmt) is __has_include-guarded, rapidhash/triple_buffer
# are vendored, and control parsing is a self-contained strtod scan. So the
# object compiles identically against score and standalone; at runtime it simply
# no-ops when the librediffusion shared library is absent (e.g. a non-CUDA host).
add_library(librediffusion_deps INTERFACE)
# 3rdparty/librediffusion/src: the StreamDiffusion C API header.
# CMAKE_CURRENT_SOURCE_DIR: the addon root, so the generated standalone back-end
# wrappers can resolve their <LibreDiffusion/LibreDiffusion.hpp> include.
target_include_directories(librediffusion_deps INTERFACE
  3rdparty/librediffusion/src
  "${CMAKE_CURRENT_SOURCE_DIR}")

# Standalone back-ends emitted for the object. Ignored in a score build (score
# introspects the object directly); only the standalone / portability back-ends
# consume this list. StreamDiffusion is an image/texture processor, so we emit
# the texture-appropriate set (cf. avendish's avnd_make_texture preset): max
# (Jitter matrix), gstreamer/touchdesigner/godot as texture filters, plus python.
# `dump` is a headless canary that keeps the object compiling in the portability
# smoke, where the SDK-backed back-ends are gated off.
set(LO_STANDALONE_BACKENDS
  dump
  python
  max
  gstreamer:TEXTURE
  touchdesigner:TOP
  godot:TEXTURE)

avnd_addon_init(NAME score_addon_librediffusion)

avnd_addon_object(
  BASE score_addon_librediffusion C_NAME streamdiffusion CLASS StreamDiffusion NAMESPACE lo
  DEPENDENCIES librediffusion_deps
  BACKENDS ${LO_STANDALONE_BACKENDS}
  SOURCES
    LibreDiffusion/LibreDiffusion.hpp
    LibreDiffusion/LibreDiffusion.cpp
    LibreDiffusion/EngineCache.hpp
    LibreDiffusion/EngineCache.cpp)

avnd_addon_finalize(NAME score_addon_librediffusion UUID b3fe4d80-0a1a-4b23-8cd2-b82472f9f91f VERSION 1)
