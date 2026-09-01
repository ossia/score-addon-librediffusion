cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(score_addon_librediffusion LANGUAGES CXX)

# Max/MSP externals link the static CRT (/MT) on Windows, as the official max-sdk-base requires.
# Only for the standalone build: inside a score build the Max backend is not emitted and score
# links the dynamic CRT, so forcing /MT here would make the addon mismatch the rest of score.
if(MSVC AND NOT TARGET score_lib_base)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# CUDA runtime only: Windows / Linux. Register a no-op install rule before bailing out so the
# score-addon CI's `cmake --build --target install` is a clean success on macOS.
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
    GIT_TAG ec179e3b22197079239f4413727589ca298c7b17)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

if(COMMAND score_common_setup)
  score_common_setup()
endif()

set(LRD_CORE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librediffusion")

# The object loads the librediffusion runtime through its C API by dynamic loading, so it needs no
# CUDA/TensorRT at build time -- only the C API header. It has no hard score dependency either
# (rapidhash / triple_buffer are vendored under LibreDiffusion/compat), so it compiles identically
# against score and standalone, and no-ops at runtime when the shared library is absent.
add_library(librediffusion_deps INTERFACE)
target_include_directories(librediffusion_deps INTERFACE
  "${LRD_CORE}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/LibreDiffusion")

# ---- Embedded engine exporter -------------------------------------------------------------------
# The plugin carries the tools that turn a HuggingFace model into a TensorRT bundle: a pinned uv
# release and librediffusion's train-lora.py with its Python package. They are turned into string
# literals by cmake/EmbedFiles.cmake and extracted at runtime by ModelBuilder.
option(LRD_EMBED_EXPORTER "Embed uv + train-lora.py so the node can build engine bundles" ON)
if(LRD_EMBED_EXPORTER AND MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # cl.exe caps a string literal at 64 KiB after concatenation (C1091); clang-cl and clang do not.
  message(WARNING "score-addon-librediffusion: the embedded exporter needs clang or clang-cl; disabled with cl.exe")
  set(LRD_EMBED_EXPORTER OFF)
endif()

set(LRD_EMBED_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/lrd-embed-manifest.txt")
set(LRD_EMBED_INPUTS "")
set(LRD_EMBED_ENTRIES "")

if(LRD_EMBED_EXPORTER)
  set(LRD_UV_VERSION 0.12.8)
  if(WIN32)
    set(LRD_UV_ASSET "uv-x86_64-pc-windows-msvc.zip")
    set(LRD_UV_SHA256 e07acf3f8a29fe41f9e04b799c3325cb0e0893836bb222bf102829b45c679ad6)
    set(LRD_UV_BINARY "uv.exe")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LRD_UV_ASSET "uv-aarch64-unknown-linux-gnu.tar.gz")
    set(LRD_UV_SHA256 ba8661f4fd207c8e94814191598e619b355ac10d5014e851e21eb800f9ef2b00)
    set(LRD_UV_BINARY "uv-aarch64-unknown-linux-gnu/uv")
  else()
    set(LRD_UV_ASSET "uv-x86_64-unknown-linux-gnu.tar.gz")
    set(LRD_UV_SHA256 2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff)
    set(LRD_UV_BINARY "uv-x86_64-unknown-linux-gnu/uv")
  endif()

  set(LRD_UV_DIR "${CMAKE_CURRENT_BINARY_DIR}/uv-${LRD_UV_VERSION}")
  set(LRD_UV_ARCHIVE "${LRD_UV_DIR}/${LRD_UV_ASSET}")
  set(LRD_UV_EXE "${LRD_UV_DIR}/${LRD_UV_BINARY}")
  if(NOT EXISTS "${LRD_UV_EXE}")
    message(STATUS "score-addon-librediffusion: downloading uv ${LRD_UV_VERSION} (${LRD_UV_ASSET})")
    file(DOWNLOAD
      "https://releases.astral.sh/github/uv/releases/download/${LRD_UV_VERSION}/${LRD_UV_ASSET}"
      "${LRD_UV_ARCHIVE}"
      EXPECTED_HASH "SHA256=${LRD_UV_SHA256}"
      STATUS LRD_UV_STATUS)
    list(GET LRD_UV_STATUS 0 LRD_UV_STATUS_CODE)
    if(LRD_UV_STATUS_CODE EQUAL 0)
      file(ARCHIVE_EXTRACT INPUT "${LRD_UV_ARCHIVE}" DESTINATION "${LRD_UV_DIR}")
    else()
      # An offline configure must not take the whole host build down with it.
      message(WARNING "score-addon-librediffusion: could not download uv (${LRD_UV_STATUS}); building without the embedded exporter")
      file(REMOVE "${LRD_UV_ARCHIVE}")
      set(LRD_EMBED_EXPORTER OFF)
    endif()
  endif()
endif()

if(LRD_EMBED_EXPORTER)

  # Manifest line: <relative path in the extracted exporter>|<source file>|<executable>
  get_filename_component(LRD_UV_NAME "${LRD_UV_BINARY}" NAME)
  list(APPEND LRD_EMBED_ENTRIES "${LRD_UV_NAME}|${LRD_UV_EXE}|1")
  list(APPEND LRD_EMBED_INPUTS "${LRD_UV_EXE}")

  file(GLOB_RECURSE LRD_EXPORTER_FILES CONFIGURE_DEPENDS
    "${LRD_CORE}/src/streamdiffusion/*"
    "${LRD_CORE}/tools/*.py")
  list(FILTER LRD_EXPORTER_FILES EXCLUDE REGEX "/__pycache__/")
  list(APPEND LRD_EXPORTER_FILES
    "${LRD_CORE}/train-lora.py"
    "${LRD_CORE}/pyproject.toml"
    "${LRD_CORE}/uv.lock")
  foreach(f IN LISTS LRD_EXPORTER_FILES)
    file(RELATIVE_PATH rel "${LRD_CORE}" "${f}")
    list(APPEND LRD_EMBED_ENTRIES "${rel}|${f}|0")
    list(APPEND LRD_EMBED_INPUTS "${f}")
  endforeach()
endif()

list(JOIN LRD_EMBED_ENTRIES "\n" LRD_EMBED_MANIFEST_TEXT)
# Rewritten only when its content changes: the manifest is a dependency of the (slow) embedding
# step, and a reconfigure must not regenerate and recompile the payload.
file(WRITE "${LRD_EMBED_MANIFEST}.tmp" "${LRD_EMBED_MANIFEST_TEXT}\n")
file(COPY_FILE "${LRD_EMBED_MANIFEST}.tmp" "${LRD_EMBED_MANIFEST}" ONLY_IF_DIFFERENT)
file(REMOVE "${LRD_EMBED_MANIFEST}.tmp")

set(LRD_EMBEDDED_CPP "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedExporter.cpp")
add_custom_command(
  OUTPUT "${LRD_EMBEDDED_CPP}"
  COMMAND "${CMAKE_COMMAND}"
          "-DLIST=${LRD_EMBED_MANIFEST}"
          "-DOUT=${LRD_EMBEDDED_CPP}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFiles.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFiles.cmake" "${LRD_EMBED_MANIFEST}" ${LRD_EMBED_INPUTS}
  COMMENT "Embedding the engine exporter (uv + train-lora.py)"
  VERBATIM)
# One static library holds the payload, so it is compiled once and linked into every back-end.
add_library(librediffusion_embedded_exporter STATIC
  LibreDiffusion/EmbeddedExporter.hpp
  "${LRD_EMBEDDED_CPP}")
target_include_directories(librediffusion_embedded_exporter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/LibreDiffusion")
set_target_properties(librediffusion_embedded_exporter PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(librediffusion_deps INTERFACE librediffusion_embedded_exporter)

# Standalone back-ends emitted for the object. Ignored in a score build (score introspects the
# object directly). StreamDiffusion is an image processor, so the texture-appropriate set; `dump`
# is the headless canary that keeps the object compiling where the SDK-backed back-ends are off.
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
    LibreDiffusion/AsyncFrameProducer.hpp
    LibreDiffusion/EngineCache.hpp
    LibreDiffusion/EngineCache.cpp
    LibreDiffusion/Handles.hpp
    LibreDiffusion/Handles.cpp
    LibreDiffusion/Image.hpp
    LibreDiffusion/ModelBuilder.hpp
    LibreDiffusion/ModelBuilder.cpp
    LibreDiffusion/Process.hpp
    LibreDiffusion/Process.cpp)

if(COMMAND score_register_license)
  score_register_license(librediffusion URL "https://github.com/jcelerier/librediffusion" FILES "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librediffusion/LICENSE")
endif()

avnd_addon_finalize(NAME score_addon_librediffusion UUID b3fe4d80-0a1a-4b23-8cd2-b82472f9f91f VERSION 1)
