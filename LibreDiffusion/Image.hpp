#pragma once

// Qt-free RGBA8888 image: an owning byte buffer + a bilinear resize. Replaces
// the QImage usage in the StreamDiffusion object so the addon builds standalone
// (pd / max / godot / ...), where Qt is not linked. The resize mirrors the
// bilinear, edge-clamped sampler the onnx addon uses for its live image path
// (score-addon-onnx ImageOps warpAffine), which is a quality upgrade over
// QImage's nearest-neighbour FastTransformation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lo
{
struct image_size
{
  int w{};
  int h{};
  int width() const noexcept { return w; }
  int height() const noexcept { return h; }
  friend bool operator==(image_size a, image_size b) noexcept
  {
    return a.w == b.w && a.h == b.h;
  }
};

struct rgba_image
{
  std::vector<unsigned char> px; // tightly packed RGBA8888 (stride == w*4)
  int w{};
  int h{};

  rgba_image() noexcept = default;

  rgba_image(int width, int height)
      : px(static_cast<std::size_t>(width) * height * 4, 0u)
      , w(width)
      , h(height)
  {
  }

  rgba_image(const unsigned char* src, int width, int height)
      : px(src, src + static_cast<std::size_t>(width) * height * 4)
      , w(width)
      , h(height)
  {
  }

  bool isNull() const noexcept { return w <= 0 || h <= 0; }
  int width() const noexcept { return w; }
  int height() const noexcept { return h; }
  image_size size() const noexcept { return {w, h}; }

  unsigned char* bits() noexcept { return px.data(); }
  const unsigned char* constBits() const noexcept { return px.data(); }
  std::size_t sizeInBytes() const noexcept { return px.size(); }

  // Bilinear, edge-clamped resize to dst.w x dst.h (RGBA8888).
  rgba_image scaled(image_size dst) const
  {
    rgba_image out(dst.w, dst.h);
    if(isNull() || dst.w <= 0 || dst.h <= 0)
      return out;

    const int sw1 = w - 1, sh1 = h - 1;
    const float scale_x = static_cast<float>(w) / dst.w;
    const float scale_y = static_cast<float>(h) / dst.h;
    for(int y = 0; y < dst.h; ++y)
    {
      const float sy = (y + 0.5f) * scale_y - 0.5f;
      const int y0 = static_cast<int>(std::floor(sy));
      const float dy = sy - y0;
      const std::size_t r0 = static_cast<std::size_t>(std::clamp(y0, 0, sh1)) * w * 4;
      const std::size_t r1
          = static_cast<std::size_t>(std::clamp(y0 + 1, 0, sh1)) * w * 4;
      unsigned char* dp = out.px.data() + static_cast<std::size_t>(y) * dst.w * 4;
      for(int x = 0; x < dst.w; ++x)
      {
        const float sx = (x + 0.5f) * scale_x - 0.5f;
        const int x0 = static_cast<int>(std::floor(sx));
        const float dx = sx - x0;
        const std::size_t c0 = static_cast<std::size_t>(std::clamp(x0, 0, sw1)) * 4;
        const std::size_t c1 = static_cast<std::size_t>(std::clamp(x0 + 1, 0, sw1)) * 4;
        for(int c = 0; c < 4; ++c)
        {
          const float v00 = px[r0 + c0 + c], v10 = px[r0 + c1 + c];
          const float v01 = px[r1 + c0 + c], v11 = px[r1 + c1 + c];
          const float top = v00 + (v10 - v00) * dx;
          const float bot = v01 + (v11 - v01) * dx;
          dp[x * 4 + c]
              = static_cast<unsigned char>(top + (bot - top) * dy + 0.5f);
        }
      }
    }
    return out;
  }
};
}
