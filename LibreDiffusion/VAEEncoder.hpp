#pragma once

#include <halp/buffer.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <stream_diffusion.hpp>
#include <tensorrt_wrappers.hpp>
namespace lo
{
struct VAEEncoder
{
  halp_meta(name, "VAE Encoder")
  halp_meta(author, "ossia team")
  halp_meta(category, "AI")
  halp_meta(manual_url, "https://ossia.io/score-docs/processes/librediffusion-generator.html")
  halp_meta(c_name, "librediffusion_vae_encoder")
  halp_meta(uuid, "8c71023a-abf7-4268-89ee-b8765b955986")

  struct
  {
    halp::texture_input<"Image", halp::rgb_texture> image;
  } inputs;

  struct
  {
    halp::texture_output<"Latent", halp::rgba32f_texture> image;
  } outputs;

  VAEEncoder();
  ~VAEEncoder();
  void operator()();

private:
  std::unique_ptr<streamdiffusion::VAEEncoderWrapper> encoder;
  std::vector<float> input_buffer;
  std::vector<float> output_buffer;
};

struct VAEDecoder
{
  halp_meta(name, "VAE Decoder")
  halp_meta(author, "ossia team")
  halp_meta(category, "AI")
  halp_meta(
      manual_url, "https://ossia.io/score-docs/processes/librediffusion-generator.html")
  halp_meta(c_name, "librediffusion_vae_decoder")
  halp_meta(uuid, "1d81f73c-aa9c-4298-965c-525780209953")

  struct
  {
    halp::texture_input<"Latent", halp::rgba32f_texture> image;
  } inputs;

  struct
  {
    halp::texture_output<"Image", halp::rgb_texture> image;
  } outputs;

  VAEDecoder();
  ~VAEDecoder();
  void operator()();

private:
  std::unique_ptr<streamdiffusion::VAEDecoderWrapper> decoder;
  std::vector<float> input_buffer;
  std::vector<float> output_buffer;
};

struct UNet
{
  halp_meta(name, "UNet")
  halp_meta(author, "ossia team")
  halp_meta(category, "AI")
  halp_meta(
      manual_url, "https://ossia.io/score-docs/processes/librediffusion-generator.html")
  halp_meta(c_name, "librediffusion_vae_decoder")
  halp_meta(uuid, "cd084ae3-d653-4f96-8302-16e7b473cfeb")

  struct
  {
    halp::texture_input<"Latent", halp::rgba32f_texture> image;
    halp::gpu_buffer_input<"Clip"> embeddings;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::rgba32f_texture> image;
  } outputs;

  UNet();
  ~UNet();
  void operator()();

private:
  std::unique_ptr<streamdiffusion::UNetWrapper> decoder;
  std::vector<float> input_buffer;
  std::vector<float> output_buffer;
};
}
