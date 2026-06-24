#pragma once
// Vendored, self-contained copy of ossia::triple_buffer (from libossia,
// ossia/detail/triple_buffer.hpp). Used only for standalone builds, where
// libossia is not on the include path; the score build picks the real header
// via __has_include in AsyncFrameProducer.hpp. Keep API-compatible with ossia.
#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ossia
{

template <typename T>
class triple_buffer
{
  static_assert(std::is_nothrow_swappable_v<T>, "T must be nothrow swappable");

  // State encoding (packed into one atomic uint8_t):
  //   bits [0:1] - middle buffer index (0, 1, or 2)
  //   bit  [2]   - dirty flag (1 = producer has written new data)
  //
  // The producer owns a thread-local `write_idx` and the consumer owns a
  // thread-local `read_idx`. The only shared atomic is `mid_state`.
  //
  // Producer: write to slot[write_idx], then exchange(write_idx | DIRTY).
  // Consumer: if dirty, exchange(read_idx | 0), then read from new read slot.

private:
  static constexpr uint8_t INDEX_MASK = 0x03;
  static constexpr uint8_t DIRTY_BIT = 0x04;

  struct alignas(64) Slot
  {
    T data;

    Slot() = default;

    template <typename U>
    explicit Slot(U&& val)
        : data(std::forward<U>(val))
    {
    }
  };

  std::array<Slot, 3> buffers_;

  alignas(64) std::atomic<uint8_t> mid_state_;
  alignas(64) uint8_t write_idx_;
  alignas(64) uint8_t read_idx_;

  triple_buffer(const triple_buffer&) = delete;
  triple_buffer& operator=(const triple_buffer&) = delete;
  triple_buffer(triple_buffer&&) = delete;
  triple_buffer& operator=(triple_buffer&&) = delete;

public:
  explicit triple_buffer(const T& init) noexcept(std::is_nothrow_copy_constructible_v<T>)
      : buffers_{Slot{init}, Slot{init}, Slot{init}}
      , mid_state_{1}
      , write_idx_{0}
      , read_idx_{2}
  {
  }

  triple_buffer() noexcept(std::is_nothrow_default_constructible_v<T>)
    requires std::is_default_constructible_v<T>
      : buffers_{}
      , mid_state_{1}
      , write_idx_{0}
      , read_idx_{2}
  {
  }

  void produce(T& value) noexcept
  {
    using std::swap;
    swap(buffers_[write_idx_].data, value);

    const uint8_t new_mid = static_cast<uint8_t>(write_idx_ | DIRTY_BIT);
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    write_idx_ = old_mid & INDEX_MASK;
  }

  void produce(T&& value) noexcept(std::is_nothrow_move_assignable_v<T>)
  {
    buffers_[write_idx_].data = std::move(value);

    const uint8_t new_mid = static_cast<uint8_t>(write_idx_ | DIRTY_BIT);
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    write_idx_ = old_mid & INDEX_MASK;
  }

  // Consume latest data if available.  Returns true and updates `result`
  // if new data was produced since the last consume(); false otherwise.
  bool consume(T& result) noexcept
  {
    const uint8_t state = mid_state_.load(std::memory_order_acquire);
    if(!(state & DIRTY_BIT))
    {
      return false;
    }

    const uint8_t new_mid = read_idx_;
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    read_idx_ = old_mid & INDEX_MASK;

    using std::swap;
    swap(result, buffers_[read_idx_].data);
    return true;
  }

  // Return a const reference to the consumer's current slot.
  // Only available for copyable types.
  //
  // Note: after consume(), the slot holds the value that was swapped *in*
  // from the caller's `result`.  For a stable "last consumed value",
  // keep your own copy.
  const T& read_buffer() const noexcept
    requires std::is_copy_assignable_v<T>
  {
    return buffers_[read_idx_].data;
  }

  bool has_new_data() const noexcept
  {
    return mid_state_.load(std::memory_order_acquire) & DIRTY_BIT;
  }
};

template <typename T>
  requires std::is_trivially_copyable_v<T>
class triple_buffer<T>
{
private:
  static constexpr uint8_t INDEX_MASK = 0x03;
  static constexpr uint8_t DIRTY_BIT = 0x04;

  struct alignas(64) Slot
  {
    T data{};
  };

  std::array<Slot, 3> buffers_;

  alignas(64) std::atomic<uint8_t> mid_state_{1};
  alignas(64) uint8_t write_idx_{0};
  alignas(64) uint8_t read_idx_{2};

  // Cached copy of last consumed value so read() is always stable.
  alignas(64) T last_read_{};

  triple_buffer(const triple_buffer&) = delete;
  triple_buffer& operator=(const triple_buffer&) = delete;
  triple_buffer(triple_buffer&&) = delete;
  triple_buffer& operator=(triple_buffer&&) = delete;

public:
  explicit triple_buffer(T init) noexcept
      : buffers_{Slot{init}, Slot{init}, Slot{init}}
      , mid_state_{1}
      , write_idx_{0}
      , read_idx_{2}
      , last_read_{init}
  {
  }

  triple_buffer() noexcept = default;

  void produce(T value) noexcept
  {
    buffers_[write_idx_].data = value;

    const uint8_t new_mid = static_cast<uint8_t>(write_idx_ | DIRTY_BIT);
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    write_idx_ = old_mid & INDEX_MASK;
  }

  // Consume latest data if available.  Returns true and updates `result`
  //with the newest value; false if no new data since last consume().
  bool consume(T& result) noexcept
  {
    const uint8_t state = mid_state_.load(std::memory_order_acquire);
    if(!(state & DIRTY_BIT))
    {
      return false;
    }

    const uint8_t new_mid = read_idx_;
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    read_idx_ = old_mid & INDEX_MASK;

    last_read_ = buffers_[read_idx_].data;
    result = last_read_;
    return true;
  }

  // Always valid
  T read() const noexcept { return last_read_; }

  bool has_new_data() const noexcept
  {
    return mid_state_.load(std::memory_order_acquire) & DIRTY_BIT;
  }
};

template <typename T, typename Container = std::vector<T>>
class triple_buffer_raw
{
  static_assert(
      std::is_nothrow_swappable_v<Container>, "Container must be nothrow swappable");

private:
  static constexpr uint8_t INDEX_MASK = 0x03;
  static constexpr uint8_t DIRTY_BIT = 0x04;

  struct alignas(64) Slot
  {
    Container data;

    Slot() = default;

    template <typename U>
    explicit Slot(U&& val)
        : data(std::forward<U>(val))
    {
    }
  };

  std::array<Slot, 3> buffers_;

  alignas(64) std::atomic<uint8_t> mid_state_;
  alignas(64) uint8_t write_idx_;
  alignas(64) uint8_t read_idx_;

  triple_buffer_raw(const triple_buffer_raw&) = delete;
  triple_buffer_raw& operator=(const triple_buffer_raw&) = delete;
  triple_buffer_raw(triple_buffer_raw&&) = delete;
  triple_buffer_raw& operator=(triple_buffer_raw&&) = delete;

public:
  triple_buffer_raw() noexcept(std::is_nothrow_default_constructible_v<Container>)
      : buffers_{}
      , mid_state_{1}
      , write_idx_{0}
      , read_idx_{2}
  {
  }

  // Pre-allocate each slot with a given capacity.
  explicit triple_buffer_raw(std::size_t initial_capacity)
      : buffers_{}
      , mid_state_{1}
      , write_idx_{0}
      , read_idx_{2}
  {
    for(auto& slot : buffers_)
      slot.data.reserve(initial_capacity);
  }

  // Direct access to the write slot. The producer writes into this
  // container however it likes, then calls publish().
  Container& write_buffer() noexcept { return buffers_[write_idx_].data; }

  // Publish the current write buffer: swap it into the middle slot
  // and pick up a new (stale) write slot.
  void publish() noexcept
  {
    const uint8_t new_mid = static_cast<uint8_t>(write_idx_ | DIRTY_BIT);
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    write_idx_ = old_mid & INDEX_MASK;
  }

  // Convenience: assign from an iterator range, then publish.
  template <typename InputIt>
  void produce(InputIt first, InputIt last)
  {
    buffers_[write_idx_].data.assign(first, last);
    publish();
  }

  // Convenience: assign from a span, then publish.
  void produce(std::span<const T> src)
  {
    buffers_[write_idx_].data.assign(src.begin(), src.end());
    publish();
  }

  // Attempt to consume. Returns true if new data was swapped in.
  // After a successful consume(), read_span() / read_buffer()
  // reflect the new data.
  bool consume() noexcept
  {
    const uint8_t state = mid_state_.load(std::memory_order_acquire);
    if(!(state & DIRTY_BIT))
      return false;

    const uint8_t new_mid = read_idx_;
    const uint8_t old_mid = mid_state_.exchange(new_mid, std::memory_order_acq_rel);
    read_idx_ = old_mid & INDEX_MASK;
    return true;
  }

  // View of the consumer's current slot.
  std::span<const T> read_span() const noexcept
  {
    const auto& c = buffers_[read_idx_].data;
    return {c.data(), c.size()};
  }

  const Container& read_buffer() const noexcept { return buffers_[read_idx_].data; }

  bool has_new_data() const noexcept
  {
    return mid_state_.load(std::memory_order_acquire) & DIRTY_BIT;
  }
};
}
