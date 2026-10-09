#include "Handles.hpp"

namespace lo
{

namespace
{
librediffusion_config_handle create_config()
{
  librediffusion_config_handle h{};
  lib().config_create(&h);
  return h;
}

// A pipeline whose engines failed to load is no pipeline.
librediffusion_pipeline_handle create_pipeline(librediffusion_config_handle config)
{
  if(!config)
    return nullptr;
  librediffusion_pipeline_handle h{};
  lib().pipeline_create(config, &h);
  if(h && lib().pipeline_init_all(h) != LIBREDIFFUSION_SUCCESS)
  {
    lib().pipeline_destroy(h);
    h = nullptr;
  }
  return h;
}

librediffusion_clip_handle create_clip(const char* engine_path, int device)
{
  librediffusion_clip_handle h{};
  lib().clip_create(engine_path, device, &h);
  return h;
}
}

SDConfig::SDConfig()
    : handle_owner{create_config()}
{
}

SDPipeline::SDPipeline(librediffusion_config_handle config)
    : handle_owner{create_pipeline(config)}
{
}

SDClip::SDClip(const char* engine_path, int device)
    : handle_owner{create_clip(engine_path, device)}
{
}

SDFluxStream::SDFluxStream(
    const char* transformer, const char* qwen, const char* vae_decoder,
    const char* vae_encoder, const char* tokenizer_json, int Th, int Tw,
    unsigned long long seed, int device)
    : handle_owner{lib().flux2_stream_create(
          transformer, qwen, vae_decoder, vae_encoder, tokenizer_json, Th, Tw, seed, device)}
{
}

SDRife::SDRife(const char* engine_path, int device)
    : handle_owner{lib().rife_create(engine_path, device)}
{
}

SDImg2ImgTurbo::SDImg2ImgTurbo(
    const char* unet, const char* vae_encoder, const char* vae_decoder, int device)
    : handle_owner{lib().img2img_turbo_create(unet, vae_encoder, vae_decoder, device)}
{
}

SDXLEmbeddings::SDXLEmbeddings(SDXLEmbeddings&& other) noexcept
    : embeddings{std::exchange(other.embeddings, nullptr)}
    , pooled_embeds{std::exchange(other.pooled_embeds, nullptr)}
    , time_ids{std::exchange(other.time_ids, nullptr)}
{
}

SDXLEmbeddings& SDXLEmbeddings::operator=(SDXLEmbeddings&& other) noexcept
{
  if(this != &other)
  {
    reset();
    embeddings = std::exchange(other.embeddings, nullptr);
    pooled_embeds = std::exchange(other.pooled_embeds, nullptr);
    time_ids = std::exchange(other.time_ids, nullptr);
  }
  return *this;
}

void SDXLEmbeddings::reset() noexcept
{
  for(auto* p : {&embeddings, &pooled_embeds, &time_ids})
  {
    if(*p)
    {
      lib().cuda_free(*p);
      *p = nullptr;
    }
  }
}

}
