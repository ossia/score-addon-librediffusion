cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

if(NOT TARGET score_plugin_avnd)
  return()
endif()

# Addon requires CUDA Toolkit
if((NOT WIN32) AND (NOT LINUX))
  return()
endif()

project(score_addon_librediffusion LANGUAGES CXX)

score_common_setup()

avnd_score_plugin_init(
  BASE_TARGET score_addon_librediffusion
)

if(0)
avnd_score_plugin_add(
  BASE_TARGET score_addon_librediffusion
  SOURCES
    LibreDiffusion/VAEEncoder.hpp
    LibreDiffusion/VAEEncoder.cpp
  TARGET librediffusion_vae_encoder
  MAIN_CLASS VAEEncoder
  NAMESPACE lo
)
avnd_score_plugin_add(
  BASE_TARGET score_addon_librediffusion
  SOURCES
    LibreDiffusion/VAEEncoder.hpp
    LibreDiffusion/VAEEncoder.cpp
  TARGET librediffusion_vae_decoder 
  MAIN_CLASS VAEDecoder
  NAMESPACE lo
)
endif()

avnd_score_plugin_add(
  BASE_TARGET score_addon_librediffusion
  SOURCES
    LibreDiffusion/LibreDiffusion.hpp
    LibreDiffusion/LibreDiffusion.cpp
    LibreDiffusion/EngineCache.hpp
    LibreDiffusion/EngineCache.cpp
  TARGET streamdiffusion
  MAIN_CLASS StreamDiffusion
  NAMESPACE lo
)


target_include_directories(score_addon_librediffusion PRIVATE 3rdparty/librediffusion/src)
avnd_score_plugin_finalize(
  BASE_TARGET score_addon_librediffusion
  PLUGIN_VERSION 1
  PLUGIN_UUID "b3fe4d80-0a1a-4b23-8cd2-b82472f9f91f"
)

