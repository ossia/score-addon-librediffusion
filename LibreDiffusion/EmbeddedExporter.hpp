#pragma once

// The engine exporter shipped inside the plugin: the uv binary plus librediffusion's
// train-lora.py and its Python package. The table is generated at build time by
// cmake/EmbedFiles.cmake; ModelBuilder extracts it under the Python cache and runs it.

#include <span>
#include <string_view>

namespace lo
{
struct EmbeddedFile
{
  std::string_view path;  // relative, '/'-separated
  std::string_view bytes;
  bool executable;
};

std::span<const EmbeddedFile> embedded_exporter_files() noexcept;

// Hash of every embedded file; names the extraction directory so a rebuilt plugin never runs a
// stale copy. Empty when the exporter was not embedded (LRD_EMBED_EXPORTER=OFF).
std::string_view embedded_exporter_id() noexcept;
}
