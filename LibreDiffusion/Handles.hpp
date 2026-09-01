#pragma once

// RAII owners for the librediffusion C API handles. Every handle is bound to the CUDA device it
// was created on (the library scopes cudaSetDevice per call), so a device change means a new
// handle, never a mutation.

#include "librediffusion_loader.hpp"

#include <utility>

namespace lo
{
inline const sd::liblibrediffusion& lib() noexcept
{
  return sd::liblibrediffusion::instance();
}

// Move-only owner of a C handle, destroyed through the loader's `Destroy` entry point
// (a pointer to one of sd::liblibrediffusion's function-pointer members).
template <typename Handle, auto Destroy>
class handle_owner
{
public:
  handle_owner() = default;
  explicit handle_owner(Handle h) noexcept
      : m_handle{h}
  {
  }
  ~handle_owner() { reset(); }
  handle_owner(const handle_owner&) = delete;
  handle_owner& operator=(const handle_owner&) = delete;
  handle_owner(handle_owner&& other) noexcept
      : m_handle{std::exchange(other.m_handle, nullptr)}
  {
  }
  handle_owner& operator=(handle_owner&& other) noexcept
  {
    if(this != &other)
    {
      reset();
      m_handle = std::exchange(other.m_handle, nullptr);
    }
    return *this;
  }

  explicit operator bool() const noexcept { return m_handle != nullptr; }
  Handle get() const noexcept { return m_handle; }

  void reset() noexcept
  {
    if(m_handle)
    {
      (lib().*Destroy)(m_handle);
      m_handle = nullptr;
    }
  }

private:
  Handle m_handle{nullptr};
};

struct SDConfig
    : handle_owner<librediffusion_config_handle, &sd::liblibrediffusion::config_destroy>
{
  SDConfig();
};

struct SDPipeline
    : handle_owner<librediffusion_pipeline_handle, &sd::liblibrediffusion::pipeline_destroy>
{
  SDPipeline() = default;
  // Creates the pipeline and loads its engines (pipeline_init_all).
  explicit SDPipeline(librediffusion_config_handle config);
};

struct SDClip : handle_owner<librediffusion_clip_handle, &sd::liblibrediffusion::clip_destroy>
{
  SDClip() = default;
  SDClip(const char* engine_path, int device);
};

struct SDFluxStream
    : handle_owner<
          librediffusion_flux2_stream_handle, &sd::liblibrediffusion::flux2_stream_destroy>
{
  SDFluxStream() = default;
  SDFluxStream(
      const char* transformer, const char* qwen, const char* vae_decoder,
      const char* vae_encoder, const char* tokenizer_json, int Th, int Tw,
      unsigned long long seed, int device);
};

struct SDRife : handle_owner<librediffusion_rife_handle, &sd::liblibrediffusion::rife_destroy>
{
  SDRife() = default;
  SDRife(const char* engine_path, int device);
};

struct SDImg2ImgTurbo
    : handle_owner<
          librediffusion_img2img_turbo_handle, &sd::liblibrediffusion::img2img_turbo_destroy>
{
  SDImg2ImgTurbo() = default;
  SDImg2ImgTurbo(const char* unet, const char* vae_encoder, const char* vae_decoder, int device);
};

// Device buffers produced by the CLIP encoders (pooled/time_ids are SDXL-only).
struct SDXLEmbeddings
{
  librediffusion_half_t* embeddings{nullptr};
  librediffusion_half_t* pooled_embeds{nullptr};
  librediffusion_half_t* time_ids{nullptr};

  SDXLEmbeddings() = default;
  ~SDXLEmbeddings() { reset(); }
  SDXLEmbeddings(const SDXLEmbeddings&) = delete;
  SDXLEmbeddings& operator=(const SDXLEmbeddings&) = delete;
  SDXLEmbeddings(SDXLEmbeddings&& other) noexcept;
  SDXLEmbeddings& operator=(SDXLEmbeddings&& other) noexcept;

  void reset() noexcept;
  explicit operator bool() const noexcept { return embeddings != nullptr; }
};

}
