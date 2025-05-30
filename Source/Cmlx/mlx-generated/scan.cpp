namespace mlx::core::metal {

const char* scan() {
  return R"preamble(
struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};
struct FloorDivide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
  template <>
  float operator()(float x, float y) {
    return trunc(x / y);
  }
  template <>
  half operator()(half x, half y) {
    return trunc(x / y);
  }
  template <>
  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
    return trunc(x / y);
  }
};
struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};
struct Remainder {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    return x % y;
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x % y;
  }
};
struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};
struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
        (metal::isnan(x.real) && metal::isnan(y.real) && metal::isnan(x.imag) &&
         metal::isnan(y.imag)) ||
        (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
        (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
  }
};
struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};
struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};
struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};
struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};
struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag) || metal::isnan(y.real) ||
        metal::isnan(y.imag)) {
      return metal::numeric_limits<float>::quiet_NaN();
    }
    constexpr float inf = metal::numeric_limits<float>::infinity();
    complex64_t maxval = x > y ? x : y;
    complex64_t minval = x < y ? x : y;
    if (minval.real == -inf || maxval.real == inf)
      return maxval;
    float m = metal::exp(minval.real - maxval.real);
    complex64_t dexp{
        m * metal::cos(minval.imag - maxval.imag),
        m * metal::sin(minval.imag - maxval.imag),
    };
    return maxval + log1p(dexp);
  }
};
struct Maximum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::max(x, y);
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};
struct Minimum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::min(x, y);
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
  }
};
struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};
struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x.real != y.real || x.imag != y.imag;
  }
};
struct Power {
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T base, T exp) {
    return metal::pow(base, exp);
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    auto x_theta = metal::atan2(x.imag, x.real);
    auto x_ln_r = 0.5 * metal::log(x.real * x.real + x.imag * x.imag);
    auto mag = metal::exp(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * metal::cos(phase), mag * metal::sin(phase)};
  }
};
struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};
struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  };
};
struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};
struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  };
};
struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  };
};
struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  };
};
struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  };
};
struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  };
};
struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return metal::precise::atan2(y, x);
  }
};
struct DivMod {
  template <typename T>
  metal::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};
template <typename U>
struct CumSum {
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true> T simd_scan(T val) { return simd_scan_impl(val); } template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> T simd_scan(T val) { for (int i = 1; i <= 16; i *= 2) { val = operator()(val, simd_shuffle_and_fill_up(val, init, i)); } return val; }
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true> T simd_exclusive_scan(T val) { return simd_exclusive_scan_impl(val); } template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> T simd_exclusive_scan(T val) { val = simd_scan(val); return simd_shuffle_and_fill_up(val, init, 1); }
  static constexpr constant U init = static_cast<U>(0);
  template <typename T>
  U operator()(U a, T b) {
    return a + b;
  }
  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_sum(x);
  }
  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_sum(x);
  }
};
template <typename U>
struct CumProd {
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true> T simd_scan(T val) { return simd_scan_impl(val); } template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> T simd_scan(T val) { for (int i = 1; i <= 16; i *= 2) { val = operator()(val, simd_shuffle_and_fill_up(val, init, i)); } return val; }
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true> T simd_exclusive_scan(T val) { return simd_exclusive_scan_impl(val); } template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> T simd_exclusive_scan(T val) { val = simd_scan(val); return simd_shuffle_and_fill_up(val, init, 1); }
  static constexpr constant U init = static_cast<U>(1.0f);
  template <typename T>
  U operator()(U a, T b) {
    return a * b;
  }
  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_product(x);
  }
  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_product(x);
  }
};
template <>
struct CumProd<bool> {
  static constexpr constant bool init = true;
  template <typename T>
  bool operator()(bool a, T b) {
    return a & static_cast<bool>(b);
  }
  bool simd_scan(bool x) {
    for (int i = 1; i <= 16; i *= 2) {
      bool other = simd_shuffle_and_fill_up(x, init, i);
      x &= other;
    }
    return x;
  }
  bool simd_exclusive_scan(bool x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};
template <typename U>
struct CumMax {
  static constexpr constant U init = Limits<U>::min;
  template <typename T>
  U operator()(U a, T b) {
    return (a >= b) ? a : b;
  }
  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x >= other) ? x : other;
    }
    return x;
  }
  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};
template <typename U>
struct CumMin {
  static constexpr constant U init = Limits<U>::max;
  template <typename T>
  U operator()(U a, T b) {
    return (a <= b) ? a : b;
  }
  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x <= other) ? x : other;
    }
    return x;
  }
  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};
template <typename U>
struct CumLogaddexp {
  static constexpr constant U init = Limits<U>::min;
  template <typename T>
  U operator()(U a, T b) {
    return LogAddExp{}(a, static_cast<U>(b));
  }
  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = LogAddExp{}(x, other);
    }
    return x;
  }
  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};
template <typename T, typename U, int N_READS, bool reverse>
inline void load_unsafe(U values[N_READS], const device T* input) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] = input[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = input[i];
    }
  }
}
template <typename T, typename U, int N_READS, bool reverse>
inline void load_safe(
    U values[N_READS],
    const device T* input,
    int start,
    int total,
    U init) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] =
          (start + N_READS - i - 1 < total) ? input[i] : init;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = (start + i < total) ? input[i] : init;
    }
  }
}
template <typename U, int N_READS, bool reverse>
inline void write_unsafe(U values[N_READS], device U* out) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[N_READS - i - 1];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[i];
    }
  }
}
template <typename U, int N_READS, bool reverse>
inline void write_safe(U values[N_READS], device U* out, int start, int total) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      if (start + N_READS - i - 1 < total) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (start + i < total) {
        out[i] = values[i];
      }
    }
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void contiguous_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  Op op;
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out += offset;
  uint simd_groups = lsize.x / simd_size;
  U prefix = Op::init;
  U values[N_READS];
  threadgroup U simdgroup_sums[32];
  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    uint offset = r * lsize.x * N_READS + lid.x * N_READS;
    if (reverse) {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(
            values, in + axis_size - offset - N_READS);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values,
            in + axis_size - offset - N_READS,
            offset,
            axis_size,
            Op::init);
      }
    } else {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(values, in + offset);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values, in + offset, offset, axis_size, Op::init);
      }
    }
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }
    U prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      U prev_simdgroup = op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
      simdgroup_sums[simd_lane_id] = prev_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }
    if (reverse) {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[axis_size - 1] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - 1 - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values,
              out + axis_size - offset - 1 - N_READS,
              offset + 1,
              axis_size);
        }
      }
    } else {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[0] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset + 1);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset + 1, offset + 1, axis_size);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void strided_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(U);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;
  threadgroup U read_buffer[BM * BN_pad];
  U values[n_scans];
  U prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = Op::init;
  }
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;
  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  threadgroup U* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup U* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;
  for (uint j = 0; j < axis_size; j += BM) {
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = in[index_y * stride + i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = in[index_y * stride + i];
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < n_scans; i++) {
      values[i] = op.simd_scan(values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
          for (int i = 0; i < N_READS; i++) {
            out[index_y * stride + i] = Op::init;
          }
        } else {
          for (int i = 0; i < N_READS; i++) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = Op::init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}
)preamble";
}

} // namespace mlx::core::metal
