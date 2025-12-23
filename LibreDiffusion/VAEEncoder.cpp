#include "VAEEncoder.hpp"

#include <kernels.hpp>
#include <stream_diffusion.hpp>

#include <cassert>

namespace lo
{

template <std::size_t C>
static void nchw_to_nhwc(const float* input, float* output, int h, int w)
{
  for(int i_c = 0; i_c < C; ++i_c)
  {
    int c_in_offset = i_c * (h * w);
    for(int i_h = 0; i_h < h; ++i_h)
    {
      int h_in_offset = c_in_offset + i_h * w;
      int h_out_offset = i_h * (w * C);

#pragma omp simd
      for(int i_w = 0; i_w < w; ++i_w)
      {
        int in_idx = h_in_offset + i_w;
        int out_idx = h_out_offset + i_w * C + i_c;

        output[out_idx] = input[in_idx];
      }
    }
  }
}

template <std::size_t C>
static void nhwc_to_nchw(const float* input, float* output, int h, int w)
{
  for(int i_c = 0; i_c < C; ++i_c)
  {
    int c_out_offset = i_c * (h * w);
    for(int i_h = 0; i_h < h; ++i_h)
    {
      int h_in_offset = i_h * (w * C);
      int h_out_offset = c_out_offset + i_h * w;

#pragma omp simd
      for(int i_w = 0; i_w < w; ++i_w)
      {
        int in_idx = h_in_offset + i_w * C + i_c;
        int out_idx = h_out_offset + i_w;

        output[out_idx] = input[in_idx];
      }
    }
  }
}

VAEEncoder::VAEEncoder()
{
  encoder = std::make_unique<streamdiffusion::VAEEncoderWrapper>(
      "/home/jcelerier/projets/oss/librediffusion-clean/engines_sdxs/"
      "vae_encoder.engine");
}
VAEEncoder::~VAEEncoder() { }

void VAEEncoder::operator()()
{
  auto& src = this->inputs.image.texture;
  if(!src.changed)
    return;

  auto image_height = src.height;
  auto image_width = src.width;
  auto latent_height = src.height / 8;
  auto latent_width = src.width / 8;

  const int latent_size = latent_width * latent_height * 4;

  if(latent_height < 1 || latent_width < 1)
    return;
  input_buffer.resize(image_height * image_width * 3);
  output_buffer.resize(latent_size);

  // latent is 64 x 64 x 4
  this->outputs.image.create(latent_width, latent_height);

  auto host_tensor = input_buffer.data();
  for(int h = 0; h < image_height; ++h)
  {
    for(int w = 0; w < image_width; ++w)
    {
      auto src_pixel = src.bytes + h * image_width * 3 + w * 3;
      // Store in CHW order
      host_tensor[0 * image_height * image_width + h * image_width + w]
          = src_pixel[0] / 127.5f - 1.f;
      host_tensor[1 * image_height * image_width + h * image_width + w]
          = src_pixel[1] / 127.5f - 1.f;
      host_tensor[2 * image_height * image_width + h * image_width + w]
          = src_pixel[2] / 127.5f - 1.f;
    }
  }

  __half* output_gpu_latent{};
  cudaMalloc(&output_gpu_latent, sizeof(__half) * latent_size);
  assert(output_gpu_latent);
  float* fp32_gpu_latent{};
  cudaMalloc(&fp32_gpu_latent, sizeof(float) * latent_size);
  assert(fp32_gpu_latent);
  this->encoder->encode(host_tensor, output_gpu_latent, 1, image_width, image_height, 0);

  launch_fp16_to_fp32(output_gpu_latent, fp32_gpu_latent, latent_size, 0);
  cudaMemcpy(
      output_buffer.data(), (const void*)fp32_gpu_latent, latent_size * sizeof(float),
      cudaMemcpyDeviceToHost);

  nchw_to_nhwc<4>(
      output_buffer.data(), this->outputs.image.texture.bytes, latent_height,
      latent_width);

  this->outputs.image.texture.changed = true;
  cudaFree(output_gpu_latent);
  cudaFree(fp32_gpu_latent);
}

VAEDecoder::VAEDecoder()
{
  decoder = std::make_unique<streamdiffusion::VAEDecoderWrapper>(
      "/home/jcelerier/projets/oss/librediffusion-clean/engines_sdxs/"
      "vae_decoder.engine");
}
VAEDecoder::~VAEDecoder() { }

void VAEDecoder::operator()()
{
  auto& src = this->inputs.image.texture;
  if(!src.changed)
    return;

  auto latent_height = src.height;
  auto latent_width = src.width;
  auto image_height = src.height * 8;
  auto image_width = src.width * 8;
  if(latent_height < 1 || latent_width < 1)
    return;
  const int latent_size = (src.width) * (src.height) * 4;
  const int image_size = (src.width * 8) * (src.height * 8) * 3;
  input_buffer.resize(latent_size);
  output_buffer.resize(image_size);

  // NHWC -> NCHW
  nhwc_to_nchw<4>(src.bytes, input_buffer.data(), latent_height, latent_width);

  float* f32_gpu_input_latent{};
  cudaMalloc(&f32_gpu_input_latent, sizeof(float) * latent_size);
  assert(f32_gpu_input_latent);
  __half* f16_gpu_input_latent{};
  cudaMalloc(&f16_gpu_input_latent, sizeof(__half) * latent_size);
  assert(f16_gpu_input_latent);

  cudaMemcpy(
      (void*)f32_gpu_input_latent, input_buffer.data(), latent_size * sizeof(float),
      cudaMemcpyDeviceToHost);

  launch_fp32_to_fp16(f32_gpu_input_latent, f16_gpu_input_latent, latent_size, 0);
  cudaStreamSynchronize(0);

  this->outputs.image.create(src.width * 8, src.height * 8);

  __half* f16_gpu_output_image{};
  cudaMalloc(&f16_gpu_output_image, sizeof(__half) * image_size);
  assert(f16_gpu_output_image);
  float* f32_gpu_output_image{};
  cudaMalloc(&f32_gpu_output_image, sizeof(float) * image_size);
  assert(f32_gpu_output_image);
  this->decoder->decode(
      f16_gpu_input_latent, f16_gpu_output_image, 1, src.width, src.height, 0);
  cudaStreamSynchronize(0);

  launch_fp16_to_fp32(f16_gpu_output_image, f32_gpu_output_image, image_size, 0);
  cudaStreamSynchronize(0);
  cudaMemcpy(
      output_buffer.data(), (const void*)f32_gpu_output_image,
      image_size * sizeof(float), cudaMemcpyDeviceToHost);

  auto rgb32f_out = output_buffer;
  auto rgb8_out = outputs.image.texture.bytes;

  // NCHW -> NWHC for the final image
  for(int h = 0; h < image_height; ++h)
  {
    for(int w = 0; w < image_width; ++w)
    {
      auto dst_pixel = rgb8_out + h * image_width * 3 + w * 3;
      // Store in CHW order
      dst_pixel[0]
          = (std::clamp(
                 rgb32f_out[0 * image_height * image_width + h * image_width + w], -1.f,
                 1.f)
             + 1.f)
            * 127.5f;
      dst_pixel[1]
          = (std::clamp(
                 rgb32f_out[1 * image_height * image_width + h * image_width + w], -1.f,
                 1.f)
             + 1.f)
            * 127.5f;
      dst_pixel[2]
          = (std::clamp(
                 rgb32f_out[2 * image_height * image_width + h * image_width + w], -1.f,
                 1.f)
             + 1.f)
            * 127.5f;
    }
  }

  this->outputs.image.texture.changed = true;
  cudaFree(f16_gpu_input_latent);
  cudaFree(f32_gpu_input_latent);
  cudaFree(f16_gpu_output_image);
  cudaFree(f32_gpu_output_image);
}

}
