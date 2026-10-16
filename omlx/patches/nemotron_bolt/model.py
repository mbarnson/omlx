"""NVIDIA Nemotron 3.5 Bolt for MLX: runs NVIDIA's IQ2_XXS/Q8_0/NVFP4 checkpoint as-is.

One file, two uses: oMLX ships it as ``omlx.patches.nemotron_bolt.model`` and
loads NVIDIA's ``NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`` repository natively;
other mlx-vlm hosts load it as a ``model_file`` (Trust Remote Code). It uses
NVIDIA's safetensors without changing them:

- IQ2_XXS (routed and shared experts): Metal kernels in this file.
- Q8_0 (attention, latent projections, lm_head): transcoded at load time to MLX
  8-bit affine, group 32. The transcode is exact.
- NVFP4 (Mamba in/out projections): MLX nvfp4 on the original bytes, plus the
  per-tensor global scale.
- BF16 (embeddings, router, norms, Mamba state, vision tower): as-is.

Speculative decoding: the language model captures layers for a block drafter
(NVIDIA's LiLiCorr), verifies up to SPEC_MAX tokens with matrix-unit kernels
whose rows do not depend on each other, and rolls Mamba and attention state
back exactly.

The IQ2_XXS grid table is llama.cpp's iq2xxs_grid (MIT, Copyright (c) 2023-2026
The ggml authors). The Nemotron-H / Nemotron Omni classes are vendored from
mlx-vlm (MIT).
"""

# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Third-party notices (MIT). The Nemotron-H / Nemotron Omni classes are adapted
# from mlx-vlm, Copyright (c) 2025 Prince Canuma. The IQ2_XXS grid table
# (iq2xxs_grid) is from ggml / llama.cpp, Copyright (c) 2023-2026 The ggml
# authors. For both:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_vlm.models.base import LanguageModelOutput as _LMOutput
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.nemotron_h_nano_omni import (  # noqa: F401  (registers the AutoProcessor)
    processing_nemotron_h_nano_omni,
)
from mlx_vlm.models.nemotron_h_nano_omni.config import (  # noqa: F401
    AudioConfig,
    VisionConfig,
)
from mlx_vlm.models.nemotron_h_nano_omni.vision import VisionModel
from mlx_vlm.models.ssm import compute_dt

QK_K = 256  # elements per IQ2_XXS block
IQ2_BLOCK_BYTES = 66  # fp16 d + 32 x uint16
Q8_BLOCK = 32  # elements per Q8_0 block
Q8_BLOCK_BYTES = 34  # fp16 d + 32 x int8

_HEADER = """
// kIQ2XXSGrid as floats: entry i is floats 8i..8i+7 (two float4).
constant float4 kIQ2XXSGridF[512] = {
@@GRIDF@@
};

// One 32-element sub-block of an IQ2_XXS block: 8 bytes after the fp16 d.
// Bytes 0..3: grid indices of four 8-element groups. Bytes 4..7: four 7-bit
// sign indices (bits 7l) and a 4-bit scale (bits 28..31).
struct IQ2Sub {
  uint grid_idx;
  uint signs_scale;
  float db;
};

// In memory the blocks are split into planes (see iq2xxs_planes): qs, one
// uint2 per sub-block, and d, one half per block. Same bytes, aligned loads.
inline IQ2Sub iq2_sub(uint2 q, half d) {
  IQ2Sub s;
  s.grid_idx = q.x;
  s.signs_scale = q.y;
  s.db = float(d) * (0.5f + float(q.y >> 28)) * 0.25f;
  return s;
}

// dot(sub-block, x[0..31]) without db: four 8-element groups, four lanes at
// a time. The grid comes as floats (no byte converts); signs flip the float
// sign bit of the grid values. No branches.
template <typename G>
inline float iq2_dot32(thread const IQ2Sub& s, G grid, thread const float4* xv) {
  float sum = 0.f;
  for (uint l = 0; l < 4; ++l) {
    uint gi = (s.grid_idx >> (8 * l)) & 255;
    uint s7 = (s.signs_scale >> (7 * l)) & 127;
    uint signs = s7 | ((popcount(s7) & 1) << 7);
    uint4 k0 = (uint4(signs) << uint4(31, 30, 29, 28)) & 0x80000000u;
    uint4 k1 = (uint4(signs) << uint4(27, 26, 25, 24)) & 0x80000000u;
    sum += dot(as_type<float4>(as_type<uint4>(grid[2 * gi]) ^ k0), xv[2 * l]);
    sum += dot(as_type<float4>(as_type<uint4>(grid[2 * gi + 1]) ^ k1), xv[2 * l + 1]);
  }
  return sum;
}

// Signed magnitudes of group l (elements 8l..8l+7), without db.
template <typename G>
inline void iq2_group(thread const IQ2Sub& s, uint l, G grid, thread float* v) {
  uint gi = (s.grid_idx >> (8 * l)) & 255;
  uint s7 = (s.signs_scale >> (7 * l)) & 127;
  uint signs = s7 | ((popcount(s7) & 1) << 7);
  uint4 k0 = (uint4(signs) << uint4(31, 30, 29, 28)) & 0x80000000u;
  uint4 k1 = (uint4(signs) << uint4(27, 26, 25, 24)) & 0x80000000u;
  float4 a = as_type<float4>(as_type<uint4>(grid[2 * gi]) ^ k0);
  float4 b = as_type<float4>(as_type<uint4>(grid[2 * gi + 1]) ^ k1);
  for (uint j = 0; j < 4; ++j) {
    v[j] = a[j];
    v[4 + j] = b[j];
  }
}
"""

# llama.cpp iq2xxs_grid: 256 entries of eight magnitude bytes.
_IQ2XXS_GRID = """    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908"""


def _grid_floats():
    vals = [int(v, 16) for v in _IQ2XXS_GRID.replace("\n", " ").split(",") if v.strip()]
    out = []
    for v in vals:
        b = [(v >> (8 * j)) & 255 for j in range(8)]
        out.append("float4({}, {}, {}, {}), float4({}, {}, {}, {})".format(*b))
    return ",\n".join(out)


_HEADER = _HEADER.replace("@@GRIDF@@", _grid_floats())


# Constant memory measured faster than a threadgroup copy for the lookups.
_LOAD_GRID = """
  constant float4* grid = kIQ2XXSGridF;
  const device uint2* qs = (const device uint2*)qsw;
"""

# Expand IQ2_XXS blocks to dense values: one thread per 32-element sub-block.
_DEQUANT = _LOAD_GRID + """
  uint sub = thread_position_in_grid.x;
  if (sub >= n_sub) return;
  uint blk = sub / 8, ib = sub % 8;
  IQ2Sub s = iq2_sub(qs[sub], d[blk]);
  float v[8];
  for (uint l = 0; l < 4; ++l) {
    iq2_group(s, l, grid, v);
    for (uint j = 0; j < 8; ++j)
      out[blk * 256 + ib * 32 + l * 8 + j] = static_cast<T>(s.db * v[j]);
  }
"""

# y[p, o] = sum_i x[xi[p], i] * W[ei[p], o, i]. A threadgroup computes ROWS
# rows; its SGS simdgroups split the input dimension (split-K) and reduce in
# threadgroup memory. Decode steps are latency-bound, so many small
# threadgroups beat a few long ones.
_GATHER_QMV = _LOAD_GRID + """
  threadgroup float part[SGS * ROWS];
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint p = threadgroup_position_in_grid.y;
  uint row0 = threadgroup_position_in_grid.x * ROWS;
  const device T* xr = x + size_t(xi[p]) * I;
  const device uint2* qe = qs + size_t(ei[p]) * O * (I / 32);
  const device half* de = d + size_t(ei[p]) * O * (I / 256);
  float acc[ROWS];
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (uint sb = sg * 32 + lane; sb < I / 32; sb += SGS * 32) {
    float4 xv[8];
    const device vec<T, 4>* xs = (const device vec<T, 4>*)(xr + sb * 32);
    for (uint k = 0; k < 8; ++k) {
      xv[k] = float4(xs[k]);
      if (RELU2) {  // relu(x)^2, rounded to T as a separate op would round it
        float4 r = max(xv[k], 0.f);
        xv[k] = float4(vec<T, 4>(r * r));
      }
    }
    uint blk = sb / 8;
    for (uint r = 0; r < ROWS; ++r) {
      uint o = row0 + r;
      if (o >= O) break;
      IQ2Sub s = iq2_sub(qe[size_t(o) * (I / 32) + sb], de[size_t(o) * (I / 256) + blk]);
      acc[r] += s.db * iq2_dot32(s, grid, xv);
    }
  }
  for (uint r = 0; r < ROWS; ++r) {
    float t = simd_sum(acc[r]);
    if (lane == 0) part[sg * ROWS + r] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < ROWS && row0 + lane < O) {
    float t = 0.f;
    for (uint g = 0; g < SGS; ++g) t += part[g * ROWS + lane];
    y[size_t(p) * O + row0 + lane] = static_cast<T>(t);
  }
"""


# Mamba2 selective scan, token by token in fp32: the multi-token form of
# mlx-vlm's decode SSM kernel (same update, same dt). X, B and C are read in
# place from the conv output u (b, L, RS) at columns 0, XB and XC. LPC lanes
# share one channel's state row; each lane holds Ds/LPC states in registers.
# With LPC = 32 the sums are in the stock decode kernel's order.
_SSM_SCAN = """
  constexpr int NS = Ds / LPC;
  constexpr int CPS = 32 / LPC;
  uint lane = thread_index_in_simdgroup;
  uint sub = lane % LPC, ch = lane / LPC;
  uint dd = thread_position_in_grid.y * CPS + ch;
  uint bh = thread_position_in_grid.z;
  uint h = bh % H, b = bh / H, grp = h / HPG;
  uint L = n_tokens;
  float A = -fast::exp(static_cast<float>(A_log[h]));
  float Dh_ = static_cast<float>(D[h]);
  float s[NS];  // lane sub holds states 4 * (LPC * j + sub) + i
  const device U* si = state_in + (size_t(bh) * Dh + dd) * Ds;
  for (int j = 0; j < NS / 4; ++j)
    for (int i = 0; i < 4; ++i) s[4 * j + i] = static_cast<float>(si[4 * (LPC * j + sub) + i]);
  for (uint t = 0; t < L; ++t) {
    size_t row = size_t(b) * L + t;
    const device T* ur = u + row * RS;
    float xv = static_cast<float>(ur[h * Dh + dd]);
    float dtv = static_cast<float>(dt[row * H + h]);
    float dA = fast::exp(A * dtv);
    float xdt = xv * dtv;
    const device vec<T, 4>* Bt = (const device vec<T, 4>*)(ur + XB + grp * Ds) + sub;
    const device vec<T, 4>* Ct = (const device vec<T, 4>*)(ur + XC + grp * Ds) + sub;
    float acc = 0.f;
    for (int j = 0; j < NS / 4; ++j) {
      float4 bv = float4(Bt[j * LPC]), cv = float4(Ct[j * LPC]);
      for (int i = 0; i < 4; ++i) {
        s[4 * j + i] = dA * s[4 * j + i] + xdt * bv[i];
        acc += s[4 * j + i] * cv[i];
      }
    }
    for (uint off = LPC / 2; off > 0; off /= 2) acc += simd_shuffle_xor(acc, off);
    if (sub == 0) out[(row * H + h) * Dh + dd] = static_cast<T>(acc + xv * Dh_);
  }
  device U* so = state_out + (size_t(bh) * Dh + dd) * Ds;
  for (int j = 0; j < NS / 4; ++j)
    for (int i = 0; i < 4; ++i) so[4 * (LPC * j + sub) + i] = static_cast<U>(s[4 * j + i]);
"""


# Prefill: y[p, o] = sum_i xs[p, i] * W[e(p), o, i] with simdgroup MMA. Each
# threadgroup decodes the IQ2_XXS weights of a BN-column tile into threadgroup
# memory, BK inputs at a time; nothing is expanded in device memory. Tiles are
# scheduled per expert. Rows are sorted by expert;
# offsets[e] is expert e's first row and tiles[e] its first tile. Tile t
# finds its expert by binary search. C goes from the MMA fragments straight
# to device memory, so threadgroup memory holds only the X and W tiles.
_GATHER_QMM = _LOAD_GRID + """
  constexpr uint BN = 64, BK = 32, LD = BK + 8, FR = 4, FRM = BM / 16;  // simdgroup tile: BM/2 x 32
  threadgroup bfloat Xs[BM * LD];
  threadgroup bfloat Ws[BN * LD];
  uint t = thread_index_in_threadgroup;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint tile = threadgroup_position_in_grid.y;
  if (tile >= tiles[NE]) return;
  uint lo = 0, hi = NE;  // largest e with tiles[e] <= tile
  while (hi - lo > 1) {
    uint mid = (lo + hi) / 2;
    if (tiles[mid] <= tile) lo = mid; else hi = mid;
  }
  uint e = lo;
  uint row0 = offsets[e] + (tile - tiles[e]) * BM;
  uint rows = min(uint(BM), offsets[e + 1] - row0);
  uint col0 = threadgroup_position_in_grid.x * BN;
  uint sr = (sg / 2) * (BM / 2), sc = (sg % 2) * 32;
  simdgroup_matrix<float, 8, 8> C[FRM][FR];
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < FRM; ++i)
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < FR; ++j) C[i][j] = simdgroup_matrix<float, 8, 8>(0.f);
  const device uint2* qe = qs + size_t(e) * O * (I / 32);
  const device half* de = d + size_t(e) * O * (I / 256);
  for (uint k0 = 0; k0 < I; k0 += BK) {
    for (uint q = t; q < BM * 4; q += 128) {  // X: 8 values per item
      uint r = q / 4, c = (q % 4) * 8;
      vec<bfloat, 8> v;
      for (uint j = 0; j < 8; ++j) v[j] = bfloat(0.f);
      if (r < rows) {
        const device vec<T, 4>* xr = (const device vec<T, 4>*)(x + size_t(row0 + r) * I + k0 + c);
        vec<T, 4> a = xr[0], b = xr[1];
        v = vec<bfloat, 8>(bfloat(a[0]), bfloat(a[1]), bfloat(a[2]), bfloat(a[3]),
                           bfloat(b[0]), bfloat(b[1]), bfloat(b[2]), bfloat(b[3]));
      }
      *(threadgroup vec<bfloat, 8>*)(Xs + r * LD + c) = v;
    }
    for (uint q = t; q < BN * 4; q += 128) {  // W: one 8-element group per item
      uint n = q / 4, l = q % 4, o = col0 + n;
      vec<bfloat, 8> v;
      for (uint j = 0; j < 8; ++j) v[j] = bfloat(0.f);
      if (o < O) {
        IQ2Sub s = iq2_sub(qe[size_t(o) * (I / 32) + k0 / 32], de[size_t(o) * (I / 256) + k0 / 256]);
        float g[8];
        iq2_group(s, l, grid, g);
        for (uint j = 0; j < 8; ++j) v[j] = bfloat(s.db * g[j]);
      }
      *(threadgroup vec<bfloat, 8>*)(Ws + n * LD + l * 8) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sr < rows)  // skip simdgroups whose rows are all padding
    _Pragma("clang loop unroll(full)")
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> A[FRM], B[FR];
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < FRM; ++i) simdgroup_load(A[i], Xs + (sr + 8 * i) * LD + kk, LD);
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < FR; ++j) simdgroup_load(B[j], Ws + (sc + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < FRM; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < FR; ++j) simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Fragment layout (as in MLX steel): lane holds (fm, fn) and (fm, fn + 1).
  uint qid = lane / 4;
  uint fm = (qid & 4) + ((lane / 2) % 4);
  uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < FRM; ++i) {
    uint r = sr + 8 * i + fm;
    if (r >= rows) continue;
    device T* yr = y + size_t(row0 + r) * O + col0;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < FR; ++j) {
      uint c = sc + 8 * j + fn;
      auto el = C[i][j].thread_elements();
      if (col0 + c + 1 < O + 1) {
        yr[c] = static_cast<T>(el[0]);
        yr[c + 1] = static_cast<T>(el[1]);
      }
    }
  }
"""


# MoE routing, one simdgroup per token: sigmoid scores (as mx.sigmoid), top-K
# of score + bias (ties to the lower expert id), then the selected scores
# normalized and scaled. Lane l holds experts l, l + 32, ...; no barriers.
_ROUTER_TOPK = """
  constexpr uint V = (E + 31) / 32;
  uint lane = thread_index_in_simdgroup;
  uint n = threadgroup_position_in_grid.y;
  float orig[V], sel[V];
  for (uint j = 0; j < V; ++j) {
    uint e = lane + 32 * j;
    orig[j] = 0.f;
    sel[j] = -INFINITY;
    if (e < E) {
      float g = static_cast<float>(logits[size_t(n) * E + e]);
      float y = 1.f / (1.f + metal::exp(metal::abs(g)));
      orig[j] = g < 0.f ? y : 1.f - y;
      sel[j] = orig[j] + static_cast<float>(bias[e]);
    }
  }
  uint mine = 0xFFFFFFFFu;  // lane k keeps the k-th pick
  float my_score = 0.f, den = 0.f;
  for (uint k = 0; k < K; ++k) {
    float v = -INFINITY;
    uint i = 0xFFFFFFFFu;
    for (uint j = 0; j < V; ++j)
      if (sel[j] > v) { v = sel[j]; i = lane + 32 * j; }
    for (uint off = 16; off > 0; off /= 2) {
      float ov = simd_shuffle_xor(v, off);
      uint oi = simd_shuffle_xor(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    float o = 0.f;
    if (i % 32 == lane) {
      o = orig[i / 32];
      sel[i / 32] = -INFINITY;
    }
    o = simd_shuffle(o, i % 32);
    den += o;
    if (lane == k) { mine = i; my_score = o; }
  }
  if (lane < K) {
    inds[size_t(n) * K + lane] = mine;
    scores[size_t(n) * K + lane] = my_score / (den + 1e-20f) * scale[0];
  }
"""


# Decode attention (one query token), split-K "flash decoding" for GQA. A
# threadgroup takes one KV head and one block of keys, and computes all G
# query heads of that KV head with simdgroup MMA: S = Q K^T, online softmax,
# O += P V. K and V tiles are staged through threadgroup memory (bounds
# checked), so each K/V byte is read once. Pass 2 merges the blocks.
_FLASH_DECODE_1 = """
  constexpr int BK = 32;
  uint blk = threadgroup_position_in_grid.x;
  uint kvh = threadgroup_position_in_grid.y;
  uint b = threadgroup_position_in_grid.z;
  uint tid = thread_index_in_threadgroup;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  int N = n_keys;
  int chunk = (N + NB - 1) / NB;
  chunk = (chunk + BK - 1) / BK * BK;
  int s0 = int(blk) * chunk;
  int s1 = min(N, s0 + chunk);
  threadgroup bfloat Qs[G * D];
  threadgroup bfloat Ks[BK * D];
  threadgroup bfloat Vs[BK * D];
  threadgroup float Ss[G * BK];
  threadgroup bfloat Ps[G * BK];
  threadgroup float fac[G], m_sh[G], l_sh[G];
  uint HQ = HKV * G;
  for (uint i = tid; i < G * D; i += 128)
    Qs[i] = bfloat(q[(size_t(b) * HQ + kvh * G) * D + i]);  // exact; scale in fp32 below
  if (tid < G) { m_sh[tid] = -INFINITY; l_sh[tid] = 0.f; }
  const device T* kb = keys + b * keys_strides[0] + kvh * keys_strides[1];
  const device T* vb = values + b * values_strides[0] + kvh * values_strides[1];
  size_t kss = keys_strides[2], vss = values_strides[2];
  simdgroup_matrix<float, 8, 8> O[2][4];
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) O[i][j] = simdgroup_matrix<float, 8, 8>(0.f);
  uint qid = lane / 4;
  uint fm = (qid & 4) + ((lane / 2) % 4);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t0 = s0; t0 < s1; t0 += BK) {
    for (uint i = tid; i < BK * D / 8; i += 128) {  // 8 values per item
      uint r = i / (D / 8), c = (i % (D / 8)) * 8;
      vec<bfloat, 8> kv;
      vec<bfloat, 8> vv;
      if (t0 + int(r) < s1) {
        vec<T, 8> a = *(const device vec<T, 8>*)(kb + size_t(t0 + r) * kss + c);
        vec<T, 8> bq = *(const device vec<T, 8>*)(vb + size_t(t0 + r) * vss + c);
        for (uint j = 0; j < 8; ++j) {
          kv[j] = bfloat(a[j]);
          vv[j] = bfloat(bq[j]);
        }
      } else {
        for (uint j = 0; j < 8; ++j) { kv[j] = bfloat(0.f); vv[j] = bfloat(0.f); }
      }
      *(threadgroup vec<bfloat, 8>*)(Ks + r * D + c) = kv;
      *(threadgroup vec<bfloat, 8>*)(Vs + r * D + c) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // S = Q K^T: simdgroup sg owns keys sg*8 .. sg*8+7
      simdgroup_matrix<float, 8, 8> S0(0.f), S1(0.f);
      for (uint d0 = 0; d0 < D; d0 += 8) {
        simdgroup_matrix<bfloat, 8, 8> A0, A1, Bm;
        simdgroup_load(A0, Qs + d0, D);
        simdgroup_load(A1, Qs + 8 * D + d0, D);
        simdgroup_load(Bm, Ks + sg * 8 * D + d0, D, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S0, A0, Bm, S0);
        simdgroup_multiply_accumulate(S1, A1, Bm, S1);
      }
      simdgroup_store(S0, Ss + sg * 8, BK);
      simdgroup_store(S1, Ss + 8 * BK + sg * 8, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // online softmax: thread handles row tid/8, columns (tid%8)*4 .. +3
      uint r = tid / 8, c0 = (tid % 8) * 4;
      float v[4], mloc = -INFINITY;
      for (uint j = 0; j < 4; ++j) {
        v[j] = (t0 + int(c0 + j) < s1) ? Ss[r * BK + c0 + j] * scale : -INFINITY;
        mloc = max(mloc, v[j]);
      }
      for (uint off = 4; off > 0; off /= 2) mloc = max(mloc, simd_shuffle_xor(mloc, off));
      float mold = m_sh[r], mnew = max(mold, mloc), sum = 0.f;
      for (uint j = 0; j < 4; ++j) {
        float p = v[j] == -INFINITY ? 0.f : fast::exp(v[j] - mnew);
        Ps[r * BK + c0 + j] = bfloat(p);
        sum += p;
      }
      for (uint off = 4; off > 0; off /= 2) sum += simd_shuffle_xor(sum, off);
      simdgroup_barrier(mem_flags::mem_none);
      if (tid % 8 == 0) {
        float f = mold == -INFINITY ? 0.f : fast::exp(mold - mnew);
        fac[r] = f;
        l_sh[r] = l_sh[r] * f + sum;
        m_sh[r] = mnew;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 2; ++i) {
      float f = fac[i * 8 + fm];
      for (int j = 0; j < 4; ++j) {
        auto el = O[i][j].thread_elements();
        el[0] *= f; el[1] *= f;
        O[i][j] = simdgroup_matrix<float, 8, 8>(0.f);
        O[i][j].thread_elements()[0] = el[0];
        O[i][j].thread_elements()[1] = el[1];
      }
    }
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> A0, A1, Bv;
      simdgroup_load(A0, Ps + kk, BK);
      simdgroup_load(A1, Ps + 8 * BK + kk, BK);
      for (int j = 0; j < 4; ++j) {
        simdgroup_load(Bv, Vs + kk * D + sg * 32 + j * 8, D);
        simdgroup_multiply_accumulate(O[0][j], A0, Bv, O[0][j]);
        simdgroup_multiply_accumulate(O[1][j], A1, Bv, O[1][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  for (int i = 0; i < 2; ++i) {
    uint r = i * 8 + fm;
    float l = l_sh[r];
    float inv = l > 0.f ? 1.f / l : 0.f;
    size_t h = size_t(b) * HQ + kvh * G + r;
    for (int j = 0; j < 4; ++j) {
      auto el = O[i][j].thread_elements();
      uint d = sg * 32 + j * 8 + fn;
      part[(h * NB + blk) * D + d] = el[0] * inv;
      part[(h * NB + blk) * D + d + 1] = el[1] * inv;
    }
  }
  if (tid < G) {
    size_t h = size_t(b) * HQ + kvh * G + tid;
    ms[h * NB + blk] = m_sh[tid];
    ls[h * NB + blk] = l_sh[tid];
  }
"""

_FLASH_DECODE_2 = """
  uint h = threadgroup_position_in_grid.y;  // b * HQ + head
  uint d = thread_position_in_threadgroup.x;
  float gmax = -INFINITY;
  for (uint i = 0; i < NB; ++i) gmax = max(gmax, ms[h * NB + i]);
  float den = 0.f, acc = 0.f;
  for (uint i = 0; i < NB; ++i) {
    float l = ls[h * NB + i];
    if (l > 0.f) {
      float w = l * fast::exp(ms[h * NB + i] - gmax);
      den += w;
      acc += w * part[(h * NB + i) * D + d];
    }
  }
  out[h * D + d] = static_cast<T>(acc / den);
"""


# ---------------------------------------------------------------------------
# Fused MoE decode (one token): four kernels per layer.
# A: router logits (bf16), latent fc1 (Q8), shared up (IQ2) from the normed x;
#    the last router threadgroup to finish picks the experts (arrival counter).
# B: routed up (IQ2) + shared down (IQ2, relu2 input).
# C: routed down (IQ2, relu2 input) with the expert-weighted sum.
# D: latent fc2 (Q8) + shared output + residual.
# Each rounds to T where the separate ops do.
# ---------------------------------------------------------------------------

_MOE_HEADER = r"""
inline float bf16_dot32(const device bfloat* w, thread const float4* xv) {
  const device vec<bfloat, 4>* w4 = (const device vec<bfloat, 4>*)w;
  float s = 0.f;
  for (uint k = 0; k < 8; ++k) s += dot(float4(w4[k]), xv[k]);
  return s;
}
// MLX affine 8-bit, group 32: w = s * u + b; xs = sum of the block's inputs.
inline float q8a_dot32(const device uint* w, float s, float b, thread const float4* xv, float xs) {
  float su = 0.f;
  for (uint k = 0; k < 8; ++k) su += dot(float4(as_type<uchar4>(w[k])), xv[k]);
  return s * su + b * xs;
}
inline float iq2_row32(const device uint2* q, const device half* d, uint sb, constant float4* grid,
                       thread const float4* xv) {
  IQ2Sub s = iq2_sub(q[sb], d[sb / 8]);
  return s.db * iq2_dot32(s, grid, xv);
}
template <typename T>
inline void load_x32(const device T* x, uint sb, bool relu2, thread float4* xv) {
  const device vec<T, 4>* xs = (const device vec<T, 4>*)(x + sb * 32);
  for (uint k = 0; k < 8; ++k) {
    float4 v = float4(xs[k]);
    if (relu2) { float4 r = max(v, 0.f); v = float4(vec<T, 4>(r * r)); }
    xv[k] = v;
  }
}
inline float sum32(thread const float4* xv) {
  float4 a = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
  return a.x + a.y + a.z + a.w;
}
// Top-k routing as route(), by one simdgroup.
template <uint E, uint K>
inline void moe_topk(const device coherent(device) float* logits, const device float* bias, float scale,
                     uint lane, device uint* inds, device float* scores) {
  constexpr uint V = (E + 31) / 32;
  float orig[V], sel[V];
  for (uint j = 0; j < V; ++j) {
    uint e = lane + 32 * j;
    orig[j] = 0.f; sel[j] = -INFINITY;
    if (e < E) {
      float g = logits[e];
      float y = 1.f / (1.f + metal::exp(metal::abs(g)));
      orig[j] = g < 0.f ? y : 1.f - y;
      sel[j] = orig[j] + bias[e];
    }
  }
  float den = 0.f, my = 0.f; uint mine = 0;
  for (uint k = 0; k < K; ++k) {
    float v = -INFINITY; uint i = 0xFFFFFFFFu;
    for (uint j = 0; j < V; ++j) if (sel[j] > v) { v = sel[j]; i = lane + 32 * j; }
    for (uint off = 16; off > 0; off /= 2) {
      float ov = simd_shuffle_xor(v, off); uint oi = simd_shuffle_xor(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    float o = 0.f;
    if (i % 32 == lane) { o = orig[i / 32]; sel[i / 32] = -INFINITY; }
    o = simd_shuffle(o, i % 32);
    den += o;
    if (lane == k) { mine = i; my = o; }
  }
  if (lane < K) { inds[lane] = mine; scores[lane] = my / (den + 1e-20f) * scale; }
}
"""

# Kernel A. Threadgroup = SG simdgroups splitting K, ROWS rows.
# Rows [0, E) router, [E, E + L) fc1 latent, [E + L, E + L + S) shared up.
_MOE_A = r"""
  constant float4* grid = kIQ2XXSGridF;
  threadgroup float part[SG * ROWS];
  threadgroup uint arrival;
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint tid = thread_index_in_threadgroup;
  uint row0 = threadgroup_position_in_grid.x * ROWS;
  uint seg = row0 < E ? 0 : (row0 < E + L ? 1 : 2);
  uint base = seg == 0 ? 0 : (seg == 1 ? E : E + L);
  uint lim = seg == 0 ? E : (seg == 1 ? L : S);
  device coherent(device) float* lg = (device coherent(device) float*)logits;
  float acc[ROWS];
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (uint sb = sg * 32 + lane; sb < H / 32; sb += SG * 32) {
    float4 xv[8];
    load_x32(xn, sb, false, xv);
    float xs = seg == 1 ? sum32(xv) : 0.f;
    for (uint r = 0; r < ROWS; ++r) {
      uint o = row0 + r - base;
      if (o >= lim) break;
      if (seg == 0) acc[r] += bf16_dot32(rw + size_t(o) * H + sb * 32, xv);
      else if (seg == 1) acc[r] += q8a_dot32((const device uint*)l1w + size_t(o) * (H / 4) + sb * 8,
                                             l1s[size_t(o) * (H / 32) + sb], l1b[size_t(o) * (H / 32) + sb], xv, xs);
      else acc[r] += iq2_row32((const device uint2*)suq + size_t(o) * (H / 32), sud + size_t(o) * (H / 256), sb, grid, xv);
    }
  }
  for (uint r = 0; r < ROWS; ++r) {
    float t = simd_sum(acc[r]);
    if (lane == 0) part[sg * ROWS + r] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < ROWS) {
    uint o = row0 + lane - base;
    if (o < lim) {
      float t = 0.f;
      for (uint j = 0; j < SG; ++j) t += part[j * ROWS + lane];
      if (seg == 0) lg[o] = float(static_cast<T>(t));
      else if (seg == 1) xl[o] = static_cast<T>(t);
      else hs[o] = static_cast<T>(t);
    }
  }
  if (seg != 0) return;
  // Router threadgroups arrive; the last one routes and resets the counter.
  device atomic_uint* ctr = (device atomic_uint*)counter;
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    arrival = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (arrival != (E + ROWS - 1) / ROWS - 1) return;
  if (sg == 0) moe_topk<E, K>(lg, bias, scale[0], lane, inds, scores);
  if (tid == 0) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
"""

# Kernel B. Routed up: threadgroups [0, NR); each simdgroup RR rows, lanes = the
# LAT / 32 sub-blocks. Shared down: threadgroups [NR, ...), SG simdgroups split K.
_MOE_B = r"""
  constant float4* grid = kIQ2XXSGridF;
  threadgroup float part[SG * RS];
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint tgi = threadgroup_position_in_grid.x;
  constexpr uint NR = K * F / (SG * RR);
  if (tgi < NR) {
    uint row0 = (tgi * SG + sg) * RR;  // F % RR == 0: one pair per simdgroup
    uint p = row0 / F;
    uint e = inds[p];
    const device uint2* q = (const device uint2*)fq + (size_t(e) * F + row0 - p * F) * (LAT / 32);
    const device half* dd = fd + (size_t(e) * F + row0 - p * F) * (LAT / 256);
    float acc[RR];
    for (uint r = 0; r < RR; ++r) acc[r] = 0.f;
    for (uint sb = lane; sb < LAT / 32; sb += 32) {
      float4 xv[8];
      load_x32(xl, sb, false, xv);
      for (uint r = 0; r < RR; ++r)
        acc[r] += iq2_row32(q + size_t(r) * (LAT / 32), dd + size_t(r) * (LAT / 256), sb, grid, xv);
    }
    for (uint r = 0; r < RR; ++r) {
      float t = simd_sum(acc[r]);
      if (lane == 0) h[row0 + r] = static_cast<T>(t);
    }
    return;
  }
  uint ro = (tgi - NR) * RS;
  float acc[RS];
  for (uint r = 0; r < RS; ++r) acc[r] = 0.f;
  for (uint sb = sg * 32 + lane; sb < S / 32; sb += SG * 32) {
    float4 xv[8];
    load_x32(hs, sb, true, xv);
    for (uint r = 0; r < RS; ++r) {
      uint o = ro + r;
      if (o >= H) break;
      acc[r] += iq2_row32((const device uint2*)sdq + size_t(o) * (S / 32), sdd + size_t(o) * (S / 256), sb, grid, xv);
    }
  }
  for (uint r = 0; r < RS; ++r) {
    float t = simd_sum(acc[r]);
    if (lane == 0) part[sg * RS + r] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < RS && ro + lane < H) {
    float t = 0.f;
    for (uint j = 0; j < SG; ++j) t += part[j * RS + lane];
    sd[ro + lane] = static_cast<T>(t);
  }
"""

# Kernel C. Threadgroup owns ROWS rows; simdgroups split the K experts.
# y[o] = sum_p score[p] * T(W_{e_p}[o] . relu2(h[p])), added in expert order.
_MOE_C = r"""
  constant float4* grid = kIQ2XXSGridF;
  threadgroup float pe[K * ROWS];
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint row0 = threadgroup_position_in_grid.x * ROWS;
  for (uint p = sg; p < K; p += SG) {
    uint e = inds[p];
    const device uint2* q = (const device uint2*)gq + (size_t(e) * LAT + row0) * (F / 32);
    const device half* dd = gd + (size_t(e) * LAT + row0) * (F / 256);
    float acc[ROWS];
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
    for (uint sb = lane; sb < F / 32; sb += 32) {
      float4 xv[8];
      load_x32(h + size_t(p) * F, sb, true, xv);
      for (uint r = 0; r < ROWS; ++r)
        acc[r] += iq2_row32(q + size_t(r) * (F / 32), dd + size_t(r) * (F / 256), sb, grid, xv);
    }
    for (uint r = 0; r < ROWS; ++r) {
      float t = float(static_cast<T>(simd_sum(acc[r]))) * scores[p];
      if (lane == 0) pe[p * ROWS + r] = t;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < ROWS) {
    float t = 0.f;
    for (uint p = 0; p < K; ++p) t += pe[p * ROWS + lane];
    y[row0 + lane] = static_cast<T>(t);
  }
"""

# Kernel D. Rows [0, H): out = x + T(T(W2 . y) + sd). SG simdgroups split K.
_MOE_D = r"""
  threadgroup float part[SG * ROWS];
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint row0 = threadgroup_position_in_grid.x * ROWS;
  float acc[ROWS];
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (uint sb = sg * 32 + lane; sb < LAT / 32; sb += SG * 32) {
    float4 xv[8];
    load_x32(y, sb, false, xv);
    float xs = sum32(xv);
    for (uint r = 0; r < ROWS; ++r) {
      uint o = row0 + r;
      acc[r] += q8a_dot32((const device uint*)w2 + size_t(o) * (LAT / 4) + sb * 8,
                          s2[size_t(o) * (LAT / 32) + sb], b2[size_t(o) * (LAT / 32) + sb], xv, xs);
    }
  }
  for (uint r = 0; r < ROWS; ++r) {
    float t = simd_sum(acc[r]);
    if (lane == 0) part[sg * ROWS + r] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < ROWS) {
    uint o = row0 + lane;
    float t = 0.f;
    for (uint j = 0; j < SG; ++j) t += part[j * ROWS + lane];
    T f = static_cast<T>(t);
    T m = static_cast<T>(float(f) + float(sd[o]));
    out[o] = static_cast<T>(float(x[o]) + float(m));
  }
"""


MOE_CFG = {"A": dict(ROWS=4, SG=4), "B": dict(RR=8, RS=4, SG=4), "C": dict(ROWS=2, SG=9), "D": dict(ROWS=2, SG=4)}


# ---------------------------------------------------------------------------
# Fused Mamba2 decode (one token): three kernels after the block norm.
# K1: in_proj (NVFP4) and, per output row, SiLU(gate), conv + SiLU with the
#     state shift, or softplus/clip of dt. K2: SSM step (the stock decode
#     kernel's sums) + D + gate; the last threadgroup writes the gated-RMSNorm
#     output. K3: out_proj (NVFP4) + residual.
# ---------------------------------------------------------------------------

def _e4m3_values():
    out = []
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = 0.0
        elif e == 0:
            v = s * (m / 8) * 2.0 ** -6
        else:
            v = s * (1 + m / 8) * 2.0 ** (e - 7)
        out.append(v)
    return out


_FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
_NVFP4_HEADER = (
    "constant float kFP4[16] = {" + ", ".join(f"{v:.9e}f" for v in _FP4) + "};\n"
    "constant float kE4M3[256] = {" + ", ".join(f"{v:.9e}f" for v in _e4m3_values()) + "};\n"
    r"""
// 32 NVFP4 weights (16 bytes) . 32 inputs, with their two e4m3 block scales.
inline float nvfp4_dot32(uint4 q, uchar2 sc, thread const float* xv) {
  float a0 = 0.f, a1 = 0.f;
  for (uint j = 0; j < 2; ++j) {
    uchar4 b = as_type<uchar4>(q[j]);
    for (uint i = 0; i < 4; ++i)
      a0 += kFP4[b[i] & 15] * xv[8 * j + 2 * i] + kFP4[b[i] >> 4] * xv[8 * j + 2 * i + 1];
  }
  for (uint j = 2; j < 4; ++j) {
    uchar4 b = as_type<uchar4>(q[j]);
    for (uint i = 0; i < 4; ++i)
      a1 += kFP4[b[i] & 15] * xv[8 * j + 2 * i] + kFP4[b[i] >> 4] * xv[8 * j + 2 * i + 1];
  }
  return kE4M3[sc.x] * a0 + kE4M3[sc.y] * a1;
}
// silu(x) = x * sigmoid(x), each op rounded to T as mlx's compiled ops do
template <typename T>
inline T silu_t(T x) {
  T e = static_cast<T>(metal::precise::exp(metal::abs(static_cast<float>(x))));
  T den = static_cast<T>(1.f + static_cast<float>(e));
  T sgm = static_cast<T>(1.f / static_cast<float>(den));
  if (!(static_cast<float>(x) < 0.f)) sgm = static_cast<T>(1.f - static_cast<float>(sgm));
  return static_cast<T>(static_cast<float>(x) * static_cast<float>(sgm));
}
""")

# K1: in_proj rows -> p = T(T(W x) * gs), then per row: gate rows -> T(silu(p));
# conv rows -> conv + bias + SiLU and the shifted conv state; dt rows -> clip(softplus(p + dt_bias)).
_MAMBA_K1 = r"""
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint row0 = (threadgroup_position_in_grid.x * SG + sg) * ROWS;
  if (row0 >= O) return;
  const device uint4* wq = (const device uint4*)w;
  const device uchar2* sc = (const device uchar2*)s;
  float acc[ROWS];
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (uint k = lane; k < I / 32; k += 32) {
    float xv[32];
    const device vec<T, 8>* xs = (const device vec<T, 8>*)(x + k * 32);
    for (uint j = 0; j < 4; ++j) { vec<T, 8> v = xs[j]; for (uint i = 0; i < 8; ++i) xv[8 * j + i] = float(v[i]); }
    for (uint r = 0; r < ROWS; ++r) {
      uint o = row0 + r;
      if (o >= O) break;
      acc[r] += nvfp4_dot32(wq[size_t(o) * (I / 32) + k], sc[size_t(o) * (I / 32) + k], xv);
    }
  }
  constexpr uint CD = DO - CO;
  for (uint r = 0; r < ROWS; ++r) {
    float t = simd_sum(acc[r]);
    uint o = row0 + r;
    if (lane != 0 || o >= O) continue;
    T pv = static_cast<T>(float(static_cast<T>(t)) * gs[0]);
    if (o >= GO && o < CO) {
      gsl[o - GO] = silu_t<T>(pv);
    } else if (o >= CO && o < DO) {
      uint c = o - CO;
      float a = 0.f;
      for (uint k = 0; k < KC - 1; ++k) a += float(cw[c * KC + k]) * float(cs[k * CD + c]);
      a += float(cw[c * KC + KC - 1]) * float(pv);
      T z = static_cast<T>(float(static_cast<T>(a)) + float(cb[c]));
      u[c] = silu_t<T>(z);
      for (uint k = 0; k + 1 < KC - 1; ++k) cs_out[k * CD + c] = cs[(k + 1) * CD + c];
      cs_out[(KC - 2) * CD + c] = pv;
    } else if (o >= DO) {
      float v = float(pv) + float(dtb[o - DO]);
      float mxv = max(v, 0.f), mnv = min(v, 0.f);
      float d = (mnv == -INFINITY || mxv == INFINITY) ? mxv : mxv + log1p(metal::exp(mnv - mxv));
      dt[o - DO] = clamp(d, lim[0], lim[1]);
    }
  }
"""

# K2: SSM step (stock decode kernel layout and sums) + D + gate; partial group sums
# of squares; the last threadgroup writes n = T(w * T(yg * rsqrt(mean_g + eps))).
_MAMBA_K2 = r"""
  threadgroup float red[TGD];
  threadgroup uint arrival;
  uint lane = thread_index_in_simdgroup;
  uint sgi = simdgroup_index_in_threadgroup;
  uint tid = thread_index_in_threadgroup;
  uint dd = thread_position_in_grid.y;
  uint h = thread_position_in_grid.z;
  uint grp = h / HPG;
  constexpr uint NS = Ds / 32;
  uint xc = h * Dh + dd;
  float xv = float(u[xc]);
  float dtv = dt[h];
  float A = -fast::exp(static_cast<float>(A_log[h]));
  float dA = fast::exp(A * dtv);
  float acc = 0.f;
  const device U* si = ssm + size_t(xc) * Ds;
  device U* so = ssm_out + size_t(xc) * Ds;
  for (uint i = 0; i < NS; ++i) {
    uint s_idx = NS * lane + i;
    float st = dA * static_cast<float>(si[s_idx]) + xv * dtv * float(u[DI + grp * Ds + s_idx]);
    so[s_idx] = static_cast<U>(st);
    acc += st * float(u[DI + G * Ds + grp * Ds + s_idx]);
  }
  acc = simd_sum(acc);
  T yv = static_cast<T>(acc + xv * float(static_cast<T>(Dp[h])));
  T ygv = static_cast<T>(float(gsl[xc]) * float(yv));
  if (lane == 0) { yg[xc] = ygv; red[sgi] = float(ygv) * float(ygv); }
  // partial sum of squares of this threadgroup's TGD channels (all in group grp)
  uint tgi = threadgroup_position_in_grid.z * (Dh / TGD) + threadgroup_position_in_grid.y;
  device coherent(device) float* pt = (device coherent(device) float*)partials;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.f;
    for (uint j = 0; j < TGD; ++j) t += red[j];
    pt[tgi] = t;
  }
  device atomic_uint* ctr = (device atomic_uint*)counter;
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    arrival = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  constexpr uint NTG = (Dh / TGD) * (DI / Dh);
  if (arrival != NTG - 1) return;
  threadgroup float inv[G];
  constexpr uint PER = NTG / G;  // threadgroups per group (heads per group x Dh / TGD)
  if (tid < G) {
    float t = 0.f;
    for (uint j = 0; j < PER; ++j) t += pt[tid * PER + j];
    inv[tid] = metal::precise::rsqrt(t / float(DI / G) + eps[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device coherent(device) T* ygc = (device coherent(device) T*)yg;
  for (uint c = tid; c < DI; c += 32 * TGD) {
    T nrm = static_cast<T>(float(ygc[c]) * inv[c / (DI / G)]);
    n[c] = static_cast<T>(float(nw[c]) * float(nrm));
  }
  if (tid == 0) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
"""

# K3: out = xres + T(T(T(W n)) * gs).
_MAMBA_K3 = r"""
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint row0 = (threadgroup_position_in_grid.x * SG + sg) * ROWS;
  if (row0 >= O) return;
  const device uint4* wq = (const device uint4*)w;
  const device uchar2* sc = (const device uchar2*)s;
  float acc[ROWS];
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (uint k = lane; k < I / 32; k += 32) {
    float xv[32];
    const device vec<T, 8>* xs = (const device vec<T, 8>*)(n + k * 32);
    for (uint j = 0; j < 4; ++j) { vec<T, 8> v = xs[j]; for (uint i = 0; i < 8; ++i) xv[8 * j + i] = float(v[i]); }
    for (uint r = 0; r < ROWS; ++r) {
      uint o = row0 + r;
      if (o >= O) break;
      acc[r] += nvfp4_dot32(wq[size_t(o) * (I / 32) + k], sc[size_t(o) * (I / 32) + k], xv);
    }
  }
  for (uint r = 0; r < ROWS; ++r) {
    float t = simd_sum(acc[r]);
    uint o = row0 + r;
    if (lane == 0 && o < O) {
      T m = static_cast<T>(float(static_cast<T>(t)) * gs[0]);
      out[o] = static_cast<T>(float(xres[o]) + float(m));
    }
  }
"""

MAMBA_CFG = {"K1": dict(ROWS=4, SG=2), "K3": dict(ROWS=4, SG=2), "TGD": 8}


@cache
def _kernel(name):
    if name == "dequant":
        return mx.fast.metal_kernel(
            name="bolt_iq2xxs_dequant",
            input_names=["qsw", "d", "n_sub"],
            output_names=["out"],
            source=_DEQUANT,
            header=_HEADER,
        )
    if name == "flash_decode_1":
        return mx.fast.metal_kernel(
            name="bolt_flash_decode_1", input_names=["q", "keys", "values", "n_keys", "scale"],
            output_names=["part", "ms", "ls"], source=_FLASH_DECODE_1,
            header="#include <metal_simdgroup_matrix>\n", ensure_row_contiguous=False,
        )
    if name == "flash_decode_2":
        return mx.fast.metal_kernel(
            name="bolt_flash_decode_2", input_names=["part", "ms", "ls"],
            output_names=["out"], source=_FLASH_DECODE_2,
        )
    if name.startswith("mamba_"):
        part = name[-2:].upper()
        names = {
            "K1": (["x", "w", "s", "gs", "cs", "cw", "cb", "dtb", "lim"], ["gsl", "u", "cs_out", "dt"]),
            "K2": (["u", "dt", "gsl", "A_log", "Dp", "ssm", "nw", "eps", "counter"], ["yg", "ssm_out", "partials", "n"]),
            "K3": (["n", "w", "s", "gs", "xres"], ["out"]),
        }[part]
        return mx.fast.metal_kernel(
            name="bolt_" + name, input_names=names[0], output_names=names[1],
            source=globals()["_MAMBA_" + part], header=_NVFP4_HEADER,
        )
    if name.startswith("moe_"):
        part = name[-1].upper()
        names = {
            "A": (["xn", "rw", "l1w", "l1s", "l1b", "suq", "sud", "bias", "scale", "counter"],
                  ["logits", "xl", "hs", "inds", "scores"]),
            "B": (["inds", "xl", "fq", "fd", "hs", "sdq", "sdd"], ["h", "sd"]),
            "C": (["inds", "scores", "h", "gq", "gd"], ["y"]),
            "D": (["x", "y", "w2", "s2", "b2", "sd"], ["out"]),
        }[part]
        return mx.fast.metal_kernel(
            name="bolt_" + name, input_names=names[0], output_names=names[1],
            source=globals()["_MOE_" + part], header=_HEADER + _MOE_HEADER,
        )
    if name == "mmv":
        return mx.fast.metal_kernel(
            name="bolt_mmv", input_names=["x", "w", "sc", "bi", "gscale"], output_names=["y"], source=_MMV,
            header="#include <metal_simdgroup_matrix>\n" + _HEADER + _NVFP4_HEADER + _mmv_tables(),
        )
    if name == "router_topk":
        return mx.fast.metal_kernel(
            name="bolt_router_topk", input_names=["logits", "bias", "scale"],
            output_names=["inds", "scores"], source=_ROUTER_TOPK,
        )
    if name == "ssm_scan":
        return mx.fast.metal_kernel(
            name="bolt_ssm_scan",
            input_names=["u", "A_log", "D", "dt", "state_in", "n_tokens"],
            output_names=["out", "state_out"],
            source=_SSM_SCAN,
        )
    if name == "conv":
        return mx.fast.metal_kernel(
            name="bolt_mamba_conv", input_names=["p", "state", "w", "bias", "n_tokens"],
            output_names=["out", "new_state"], source=_CONV,
        )
    if name == "gather_qmm":
        return mx.fast.metal_kernel(
            name="bolt_iq2xxs_gather_qmm",
            input_names=["x", "qsw", "d", "offsets", "tiles"],
            output_names=["y"],
            source=_GATHER_QMM,
            header="#include <metal_simdgroup_matrix>\n" + _HEADER,
        )
    return mx.fast.metal_kernel(
        name="bolt_iq2xxs_gather_qmv",
        input_names=["x", "qsw", "d", "xi", "ei"],
        output_names=["y"],
        source=_GATHER_QMV,
        header=_HEADER,
    )


def iq2xxs_planes(blocks: mx.array):
    """(..., O, NB, 66) IQ2_XXS blocks -> planes (qs, d).

    qs: (..., O, NB * 16) uint32, two words per 32-element sub-block.
    d:  (..., O, NB) float16, one scale per block.
    The bytes are the checkpoint's; only their order in memory changes.
    """
    assert blocks.shape[-1] == IQ2_BLOCK_BYTES, blocks.shape
    *lead, nb, _ = blocks.shape
    d = blocks[..., :2].view(mx.float16).reshape(*lead, nb)
    qs = blocks[..., 2:].reshape(*lead, nb * 64).view(mx.uint32)
    return qs, d


def _in_dims(d):
    return d.shape[-1] * QK_K


def iq2xxs_dequantize(qs, d, dtype=mx.bfloat16) -> mx.array:
    """Planes -> (..., O, I) dense."""
    n_sub = d.size * 8
    (out,) = _kernel("dequant")(
        inputs=[qs, d, mx.array(n_sub, dtype=mx.uint32)],
        template=[("T", dtype)],
        grid=(n_sub, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*d.shape[:-1], _in_dims(d))],
        output_dtypes=[dtype],
    )
    return out


QMV_ROWS = 4  # output rows per threadgroup


def _qmv_sgs(I):
    # One simdgroup per 1024 inputs (32 sub-blocks, one per lane), up to 4.
    return max(1, min(4, I // 1024))


def iq2xxs_gather_qmv(x, qs, d, xi, ei, rows=None, sgs=None, relu2=False):
    """x (N, I); planes of (E, O, I); xi, ei (P,) uint32 -> (P, O).
    relu2: use relu(x)^2 (rounded to x.dtype) in place of x."""
    O, I = d.shape[-2], _in_dims(d)
    assert x.shape[-1] == I, (x.shape, d.shape)
    P = xi.shape[0]
    rows, sgs = rows or QMV_ROWS, sgs or _qmv_sgs(I)
    (y,) = _kernel("gather_qmv")(
        inputs=[x, qs, d, xi, ei],
        template=[("T", x.dtype), ("I", I), ("O", O), ("ROWS", rows), ("SGS", sgs),
                  ("RELU2", bool(relu2))],
        grid=(((O + rows - 1) // rows) * 32 * sgs, P, 1),
        threadgroup=(32 * sgs, 1, 1),
        output_shapes=[(P, O)],
        output_dtypes=[x.dtype],
    )
    return y


def route(x, weight, bias, top_k, scale):
    """Sigmoid top-k routing (n_group 1). x (N, H) -> inds (N, k) uint32, scores (N, k) fp32."""
    N, H = x.shape
    E = weight.shape[0]
    logits = x @ weight.T  # as mlx-vlm's MoEGate
    assert top_k <= 32, top_k
    inds, scores = _kernel("router_topk")(
        inputs=[logits, bias, mx.array([scale], mx.float32)],
        template=[("T", logits.dtype), ("E", E), ("K", top_k)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, top_k), (N, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, scores


FLASH_DECODE_MIN_KEYS = 2048  # below this, MLX's decode attention is as fast


def flash_decode_supported(q, keys):
    B, HQ, L, D = q.shape
    return L == 1 and D == 128 and 16 * keys.shape[1] == HQ and keys.shape[-1] == 128


def flash_decode(q, keys, values, scale):
    """q (B, HQ, 1, 128); keys, values (B, HKV, N, 128), HQ = 16 HKV. Any
    strides along N (e.g. a view into a KV cache buffer); rows contiguous."""
    B, HQ, _, D = q.shape
    HKV, N = keys.shape[1], keys.shape[2]
    nb = max(1, min(128, N // 256 if N < 16384 else N // 512))
    part, ms, ls = _kernel("flash_decode_1")(
        inputs=[mx.contiguous(q), keys, values, mx.array(N, mx.int32), mx.array(scale, mx.float32)],
        template=[("T", q.dtype), ("D", D), ("G", HQ // HKV), ("HKV", HKV), ("NB", nb)],
        grid=(nb * 128, HKV, B),
        threadgroup=(128, 1, 1),
        output_shapes=[(B * HQ, nb, D), (B * HQ, nb), (B * HQ, nb)],
        output_dtypes=[mx.float32] * 3,
    )
    (out,) = _kernel("flash_decode_2")(
        inputs=[part, ms, ls],
        template=[("T", q.dtype), ("D", D), ("NB", nb)],
        grid=(D, B * HQ, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, HQ, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out


def _moe_tpl(name, T, extra):
    return [("T", T)] + list(MOE_CFG[name].items()) + extra


def _arrival_counter():
    """A zeroed arrival counter for one kernel launch.

    The last threadgroup to arrive does the cross-threadgroup step. A buffer
    shared between launches would let two independent graphs (evaluated
    together, so free to overlap) count each other's arrivals.
    More than one element, so MLX passes it as a device buffer.
    """
    return mx.zeros((32,), mx.uint32)


def moe_decode(mixer, x, xn):
    """mixer: BoltMoE; x: (1, 1, H) residual input; xn: normed x. Returns x + moe(xn)."""
    g = mixer.gate
    E, H = g.weight.shape
    f1, f2 = mixer.fc1_latent_proj, mixer.fc2_latent_proj
    L = f1.weight.shape[0]
    su, sdn = mixer.shared_experts.up_proj, mixer.shared_experts.down_proj
    S = su.weight.shape[1]
    sw = mixer.switch_mlp
    F = sw.fc1.weight.shape[1]
    k = g.top_k
    T = x.dtype
    if not hasattr(mixer, "_scale"):
        mixer._scale = mx.array([g.routed_scaling_factor], mx.float32)
        mx.eval(mixer._scale)
    ca, cb, cc, cd = MOE_CFG["A"], MOE_CFG["B"], MOE_CFG["C"], MOE_CFG["D"]
    _, xl, hs, inds, scores = _kernel("moe_a")(
        inputs=[xn.reshape(H), g.weight, f1.weight, f1.scales, f1.biases, su.weight, su.scales,
                g.e_score_correction_bias, mixer._scale, _arrival_counter()],
        template=_moe_tpl("A", T, [("H", H), ("E", E), ("L", L), ("S", S), ("K", k)]),
        grid=(((E + L + S + ca["ROWS"] - 1) // ca["ROWS"]) * 32 * ca["SG"], 1, 1), threadgroup=(32 * ca["SG"], 1, 1),
        output_shapes=[(E,), (L,), (S,), (k,), (k,)], output_dtypes=[mx.float32, T, T, mx.uint32, mx.float32])
    nr = k * F // (cb["SG"] * cb["RR"])
    h, sd = _kernel("moe_b")(
        inputs=[inds, xl, sw.fc1.weight, sw.fc1.scales, hs, sdn.weight, sdn.scales],
        template=_moe_tpl("B", T, [("H", H), ("K", k), ("F", F), ("LAT", L), ("S", S)]),
        grid=((nr + (H + cb["RS"] - 1) // cb["RS"]) * 32 * cb["SG"], 1, 1), threadgroup=(32 * cb["SG"], 1, 1),
        output_shapes=[(k * F,), (H,)], output_dtypes=[T, T])
    (y,) = _kernel("moe_c")(
        inputs=[inds, scores, h, sw.fc2.weight, sw.fc2.scales],
        template=_moe_tpl("C", T, [("K", k), ("F", F), ("LAT", L)]),
        grid=((L // cc["ROWS"]) * 32 * cc["SG"], 1, 1), threadgroup=(32 * cc["SG"], 1, 1),
        output_shapes=[(L,)], output_dtypes=[T])
    (out,) = _kernel("moe_d")(
        inputs=[x.reshape(H), y, f2.weight, f2.scales, f2.biases, sd],
        template=_moe_tpl("D", T, [("H", H), ("LAT", L)]),
        grid=((H // cd["ROWS"]) * 32 * cd["SG"], 1, 1), threadgroup=(32 * cd["SG"], 1, 1),
        output_shapes=[(H,)], output_dtypes=[T])
    return out.reshape(x.shape)


def moe_decode_supported(mixer):
    g = mixer.gate
    F = mixer.switch_mlp.fc1.weight.shape[1]
    return (
        g.n_group == 1 and g.norm_topk_prob and 1 < g.top_k <= 32
        and F % MOE_CFG["B"]["RR"] == 0 and mixer.fc1_latent_proj.weight.shape[0] % 32 == 0
    )


def mamba_decode(mixer, x, xn, conv_state, ssm_state):
    """x: (1, 1, H) residual input, xn: normed x. Returns (x + mamba(xn), conv_state, ssm_state)."""
    m = mixer
    H = x.shape[-1]
    T = x.dtype
    O1, I1 = m.in_proj.weight.shape[0], m.in_proj.weight.shape[1] * 8
    base = m.intermediate_size + m.conv_dim + m.num_heads
    GO = O1 - base
    CO = GO + m.intermediate_size
    DO = CO + m.conv_dim
    KC = m.conv_kernel_size
    if not hasattr(m, "_lim"):
        lo, hi = m.time_step_limit
        m._lim = mx.array([lo, hi], mx.float32)
        m._eps = mx.array([m.norm.eps], mx.float32)
        mx.eval(m._lim, m._eps)
    c1, c3, tgd = MAMBA_CFG["K1"], MAMBA_CFG["K3"], MAMBA_CFG["TGD"]
    gsl, u, cs, dt = _kernel("mamba_k1")(
        inputs=[xn.reshape(H), m.in_proj.weight, m.in_proj.scales, m.in_proj.global_scale.reshape(1),
                conv_state.reshape(KC - 1, m.conv_dim), m.conv1d.weight, m.conv1d.bias, m.dt_bias, m._lim],
        template=[("T", T), ("I", I1), ("O", O1), ("ROWS", c1["ROWS"]), ("SG", c1["SG"]),
                  ("GO", GO), ("CO", CO), ("DO", DO), ("KC", KC)],
        grid=(((O1 + c1["ROWS"] * c1["SG"] - 1) // (c1["ROWS"] * c1["SG"])) * 32 * c1["SG"], 1, 1),
        threadgroup=(32 * c1["SG"], 1, 1),
        output_shapes=[(m.intermediate_size,), (m.conv_dim,), (KC - 1, m.conv_dim), (m.num_heads,)],
        output_dtypes=[T, T, conv_state.dtype, mx.float32])
    ntg = (m.head_dim // tgd) * m.num_heads
    yg, ss, _, n = _kernel("mamba_k2")(
        inputs=[u, dt, gsl, m.A_log, m.D, ssm_state.reshape(m.num_heads * m.head_dim, m.ssm_state_size),
                m.norm.weight, m._eps, _arrival_counter()],
        template=[("T", T), ("U", ssm_state.dtype), ("Dh", m.head_dim), ("Ds", m.ssm_state_size), ("G", m.n_groups),
                  ("HPG", m.num_heads // m.n_groups), ("DI", m.intermediate_size), ("TGD", tgd)],
        grid=(32, m.head_dim, m.num_heads), threadgroup=(32, tgd, 1),
        output_shapes=[(m.intermediate_size,), (m.num_heads * m.head_dim, m.ssm_state_size), (ntg,), (m.intermediate_size,)],
        output_dtypes=[T, ssm_state.dtype, mx.float32, T])
    O, I = m.out_proj.weight.shape[0], m.out_proj.weight.shape[1] * 8
    (out,) = _kernel("mamba_k3")(
        inputs=[n, m.out_proj.weight, m.out_proj.scales, m.out_proj.global_scale.reshape(1), x.reshape(H)],
        template=[("T", T), ("I", I), ("O", O), ("ROWS", c3["ROWS"]), ("SG", c3["SG"])],
        grid=(((O + c3["ROWS"] * c3["SG"] - 1) // (c3["ROWS"] * c3["SG"])) * 32 * c3["SG"], 1, 1),
        threadgroup=(32 * c3["SG"], 1, 1), output_shapes=[(O,)], output_dtypes=[T])
    return out.reshape(x.shape), cs.reshape(conv_state.shape), ss.reshape(ssm_state.shape)


SSM_LPC = 4  # lanes per channel in the scan kernel


def ssm_scan(u, A_log, D, dt, state, heads, head_dim, groups, lpc=None):
    """Selective scan. u (b, L, RS): conv output, x at columns [0, H * Dh), then
    B and C (G x Ds each); dt (b, L, H) fp32 after softplus and clip; state
    (b, H, Dh, Ds). Returns y (b, L, H * Dh) and the new state."""
    b, L, rs = u.shape
    ds = state.shape[-1]
    xb = heads * head_dim
    lpc = lpc or SSM_LPC
    y, state = _kernel("ssm_scan")(
        inputs=[u, A_log, D, dt, state, mx.array(L, dtype=mx.uint32)],
        template=[("T", u.dtype), ("U", state.dtype), ("Dh", head_dim), ("Ds", ds),
                  ("H", heads), ("G", groups), ("HPG", heads // groups), ("LPC", lpc),
                  ("RS", rs), ("XB", xb), ("XC", xb + groups * ds)],
        grid=(32, head_dim * lpc // 32, heads * b),
        threadgroup=(32, 4 if lpc < 32 else 8, 1),
        output_shapes=[(b, L, heads * head_dim), state.shape],
        output_dtypes=[u.dtype, state.dtype],
    )
    return y, state


# Mamba2 causal depthwise conv + bias + SiLU over L new tokens, reading the
# conv input in place from the in_proj output p (b, L, PS) at column PO, and
# the K-1 previous inputs from the conv state. Rounds like mlx-vlm's ops:
# conv sum, + bias, and each step of x * sigmoid(x) round to T.
_CONV = """
  uint c = thread_position_in_grid.x;
  uint t = thread_position_in_grid.y;
  uint b = thread_position_in_grid.z;
  if (c >= C) return;
  uint L = n_tokens;
  float acc = 0.f;
  for (uint k = 0; k < K; ++k) {
    int src = int(t + k) - int(K - 1);  // < 0: from the state
    T v = src < 0 ? state[(size_t(b) * (K - 1) + (K - 1) + src) * C + c]
                  : p[(size_t(b) * L + src) * PS + PO + c];
    acc += static_cast<float>(w[c * K + k]) * static_cast<float>(v);
  }
  T z = static_cast<T>(static_cast<float>(static_cast<T>(acc)) + static_cast<float>(bias[c]));
  T e = static_cast<T>(metal::precise::exp(metal::abs(static_cast<float>(z))));
  T den = static_cast<T>(1.f + static_cast<float>(e));
  T sg = static_cast<T>(1.f / static_cast<float>(den));
  if (!(static_cast<float>(z) < 0.f)) sg = static_cast<T>(1.f - static_cast<float>(sg));
  out[(size_t(b) * L + t) * C + c] = static_cast<T>(static_cast<float>(z) * static_cast<float>(sg));
  if (t == 0) {  // new state: the last K-1 rows of [state; inputs]
    for (uint j = 0; j < K - 1; ++j) {
      uint i = L + j;
      new_state[(size_t(b) * (K - 1) + j) * C + c] =
          i < K - 1 ? state[(size_t(b) * (K - 1) + i) * C + c]
                    : p[(size_t(b) * L + i - (K - 1)) * PS + PO + c];
    }
  }
"""


def mamba_conv(p, offset, state, weight, bias, channels):
    """Depthwise causal conv + SiLU. p (b, L, PS) in_proj output; conv input at
    columns [offset, offset + C); state (b, K - 1, C); weight (C, K, 1)."""
    b, L, ps = p.shape
    K = weight.shape[1]
    out, new_state = _kernel("conv")(
        inputs=[p, state, weight, bias, mx.array(L, dtype=mx.uint32)],
        template=[("T", p.dtype), ("C", channels), ("K", K), ("PS", ps), ("PO", offset)],
        grid=(channels, L, b),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, L, channels), (b, K - 1, channels)],
        output_dtypes=[p.dtype, p.dtype],
    )
    return out, new_state


QMM_BM = 64


def expert_tiles(es, num_experts, bm=QMM_BM):
    """Sorted expert ids (P,) -> (offsets, tiles), each (E + 1,) uint32."""
    counts = mx.zeros((num_experts,), mx.uint32).at[es].add(mx.array(1, mx.uint32))
    zero = mx.zeros((1,), mx.uint32)
    offsets = mx.concatenate([zero, mx.cumsum(counts)])
    tiles = mx.concatenate([zero, mx.cumsum((counts + bm - 1) // bm)])
    return offsets, tiles


def iq2xxs_gather_qmm(xs, qs, d, es, tiles=None, bm=None):
    """xs (P, I) rows sorted by expert; planes of (E, O, I); es (P,) -> (P, O).
    tiles: expert_tiles(es, E), to share between calls with the same es."""
    O, I = d.shape[-2], _in_dims(d)
    E, P = d.shape[0], xs.shape[0]
    bm = bm or QMM_BM
    offsets, tiles = tiles if tiles is not None else expert_tiles(es, E, bm)
    n_tiles = min((P + bm - 1) // bm + E, P)
    (y,) = _kernel("gather_qmm")(
        inputs=[xs, qs, d, offsets, tiles],
        template=[("T", xs.dtype), ("I", I), ("O", O), ("NE", E), ("BM", bm)],
        grid=(((O + 63) // 64) * 128, n_tiles, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(P, O)],
        output_dtypes=[xs.dtype],
    )
    return y


def q8_0_to_affine(w: mx.array, scale_dtype=mx.float32):
    """(O, NB, 34) uint8 Q8_0 -> MLX affine 8-bit, group 32 (weight, scales, biases).

    Q8_0 stores w = d * q, q int8. MLX stores w = s * u + b, u uint8. With
    u = q + 128, s = d, b = -128 * d the values are the same.
    """
    O, nb, nbytes = w.shape
    assert nbytes == Q8_BLOCK_BYTES, w.shape
    d = w[..., :2].view(mx.float16).squeeze(-1)  # (O, NB)
    u = w[..., 2:] ^ mx.array(0x80, mx.uint8)  # int8 two's complement -> q + 128
    weight = u.reshape(O, nb * Q8_BLOCK).view(mx.uint32)
    scales = d.astype(scale_dtype)
    return weight, scales, -128 * scales


# ---------------------------------------------------------------------------
# Layers over the checkpoint's own tensors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Speculative verify (2..8 tokens): small-batch matmul on the matrix units.
# y[t, o] = sum_k x[t, k] W[o, k]. A simdgroup owns RB blocks of 8 output rows
# (fragment A, decoded from the stored format) against the tokens (fragment B,
# padded to 8). A lane decodes 8 consecutive weights of one row per 32-element
# sub-block; inside a sub-block, A and B use the same permuted k order. Token
# columns never mix: a token's result does not depend on the other tokens.
# ---------------------------------------------------------------------------

MMV_FMT = {"bf16": 0, "q8": 1, "iq2": 2, "nvfp4": 3}
MMV_MAX = 8  # rows (tokens) per call
MMV_CFG = dict(SG=4, RB=2)
MMV_MIN_TG = 1  # split K across threadgroups until the grid has this many (off: no gain)

_MMV = r"""
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint qid = lane / 4;
  uint fm = (qid & 4) + ((lane / 2) % 4);
  uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  uint p = fn / 2;
  uint ob = threadgroup_position_in_grid.x * (8 * RB);
  // A (weights) and B (tokens) in bf16 are exact: x is bf16; A holds raw NVFP4
  // values, signed IQ2_XXS grid values or Q8 integers. The per-row block scales
  // are applied to the fp32 partial sums after each 16- or 32-element block.
  simdgroup_matrix<float, 8, 8> C[RB], P[RB];
  simdgroup_matrix<bfloat, 8, 8> A, B;
  for (uint r = 0; r < RB; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0.f);
  for (uint sb = threadgroup_position_in_grid.y * SG + sg; sb < I / 32; sb += SG * KS) {
    uint pk[RB][4];
    float ps[RB][2];
    for (uint r = 0; r < RB; ++r) {
      uint o = min(ob + 8 * r + fm, uint(O - 1));
      if (FMT == 0) {
        uint4 v = *(const device uint4*)(w + size_t(o) * I + sb * 32 + p * 8);
        pk[r][0] = v.x; pk[r][1] = v.y; pk[r][2] = v.z; pk[r][3] = v.w;
      } else if (FMT == 1) {
        uint2 v = *(const device uint2*)((const device uint*)w + size_t(o) * (I / 4) + sb * 8 + p * 2);
        pk[r][0] = v.x; pk[r][1] = v.y; ps[r][0] = sc[size_t(o) * (I / 32) + sb];
      } else if (FMT == 2) {
        IQ2Sub s = iq2_sub(((const device uint2*)w)[size_t(o) * (I / 32) + sb], ((const device half*)sc)[size_t(o) * (I / 256) + sb / 8]);
        uint s7 = (s.signs_scale >> (7 * p)) & 127;
        pk[r][0] = ((s.grid_idx >> (8 * p)) & 255) * 4;
        pk[r][1] = s7 | ((popcount(s7) & 1) << 7);
        ps[r][0] = s.db;
      } else {
        // Lane p: nibbles 4p..4p+3 of each 16-element half (two bytes each).
        const device ushort* q = (const device ushort*)w + (size_t(o) * (I / 32) + sb) * 8 + p;
        pk[r][0] = q[0]; pk[r][1] = q[4];
        uchar2 s2 = ((const device uchar2*)sc)[size_t(o) * (I / 32) + sb];
        ps[r][0] = kE4M3[s2.x]; ps[r][1] = kE4M3[s2.y];
      }
    }
    for (uint r = 0; r < RB; ++r) P[r] = simdgroup_matrix<float, 8, 8>(0.f);
    for (uint c = 0; c < 4; ++c) {
      // k of this lane's B elements; A uses the same order (row fm, pair fn).
      uint k = FMT == 3 ? sb * 32 + (c / 2) * 16 + (fm / 2) * 4 + (c % 2) * 2 + (fm % 2)
                        : sb * 32 + (fm / 2) * 8 + 2 * c + (fm % 2);
      B.thread_elements()[0] = fn < NT ? static_cast<bfloat>(x[size_t(fn) * I + k]) : bfloat(0.f);
      B.thread_elements()[1] = fn + 1 < NT ? static_cast<bfloat>(x[size_t(fn + 1) * I + k]) : bfloat(0.f);
      for (uint r = 0; r < RB; ++r) {
        bfloat2 a;
        if (FMT == 0) {
          a = as_type<bfloat2>(pk[r][c]);
        } else if (FMT == 1) {
          uchar4 u = as_type<uchar4>(pk[r][c / 2]);
          uint j = (c % 2) * 2;
          a = bfloat2(bfloat(float(int(u[j]) - 128)), bfloat(float(int(u[j + 1]) - 128)));
        } else if (FMT == 2) {
          uint sgn = pk[r][1] >> (2 * c);
          a = as_type<bfloat2>(kIQ2BF2[pk[r][0] + c] ^ (((sgn & 1) << 15) | ((sgn & 2) << 30)));
        } else {
          a = as_type<bfloat2>(kFP4BF2[(pk[r][c / 2] >> (8 * (c % 2))) & 255]);
        }
        A.thread_elements()[0] = a.x;
        A.thread_elements()[1] = a.y;
        if (FMT == 0) simdgroup_multiply_accumulate(C[r], A, B, C[r]);
        else simdgroup_multiply_accumulate(P[r], A, B, P[r]);
      }
      if (FMT == 3 && c == 1)
        for (uint r = 0; r < RB; ++r) {
          C[r].thread_elements()[0] += ps[r][0] * P[r].thread_elements()[0];
          C[r].thread_elements()[1] += ps[r][0] * P[r].thread_elements()[1];
          P[r] = simdgroup_matrix<float, 8, 8>(0.f);
        }
    }
    if (FMT != 0)
      for (uint r = 0; r < RB; ++r) {
        float s = FMT == 3 ? ps[r][1] : ps[r][0];
        C[r].thread_elements()[0] += s * P[r].thread_elements()[0];
        C[r].thread_elements()[1] += s * P[r].thread_elements()[1];
      }
  }
  threadgroup float part[SG * RB * 64];
  for (uint r = 0; r < RB; ++r) {
    part[(sg * RB + r) * 64 + fm * 8 + fn] = C[r].thread_elements()[0];
    part[(sg * RB + r) * 64 + fm * 8 + fn + 1] = C[r].thread_elements()[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = sg * 32 + lane; e < RB * 64; e += SG * 32) {
    uint r = e / 64, rr = (e % 64) / 8, t = e % 8;
    uint o = ob + 8 * r + rr;
    if (t >= NT || o >= O) continue;
    float a = 0.f;
    for (uint j = 0; j < SG; ++j) a += part[(j * RB + r) * 64 + rr * 8 + t];
    if (KS == 1) y[size_t(t) * O + o] = static_cast<OT>(static_cast<float>(static_cast<RT>(a)) * gscale[0]);
    else y[(size_t(threadgroup_position_in_grid.y) * NT + t) * O + o] = static_cast<OT>(a);
  }
"""



def _bf16_bits(v):
    import struct
    return struct.unpack("<I", struct.pack("<f", v))[0] >> 16  # exact for these values


def _mmv_tables():
    fp4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    pairs = [_bf16_bits(fp4[b & 15]) | (_bf16_bits(fp4[b >> 4]) << 16) for b in range(256)]
    out = "constant uint kFP4BF2[256] = {" + ", ".join(f"0x{v:08x}u" for v in pairs) + "};\n"
    vals = [int(v, 16) for v in _IQ2XXS_GRID.replace("\n", " ").split(",") if v.strip()]
    g = []
    for v in vals:
        b = [(v >> (8 * j)) & 255 for j in range(8)]
        g += [_bf16_bits(float(b[2 * c])) | (_bf16_bits(float(b[2 * c + 1])) << 16) for c in range(4)]
    return out + "constant uint kIQ2BF2[1024] = {" + ", ".join(f"0x{v:08x}u" for v in g) + "};\n"




def mmv(x, fmt, w, O, I, sc=None, bi=None, out_dtype=None, gscale=None):
    """x (N <= 8, I) with the weight in its stored format -> (N, O).

    gscale (NVFP4's per-tensor scale): y = T(T(x W) * gscale), as NVFP4Linear rounds."""
    n = x.shape[0]
    dummy = _CONST.get("mmv_dummy")
    if dummy is None:
        dummy = _CONST["mmv_dummy"] = mx.zeros((64,), mx.float32)  # small arrays become constant
        _CONST["mmv_one"] = mx.ones((1,), mx.float32)
        mx.eval(dummy, _CONST["mmv_one"])
    out_dtype = out_dtype or x.dtype
    sg, rb = MMV_CFG["SG"], MMV_CFG["RB"]
    tgs = (O + 8 * rb - 1) // (8 * rb)
    ks = 1
    while tgs * ks < MMV_MIN_TG and (I // 32) % (2 * ks * sg) == 0 and I // (32 * 2 * ks * sg) >= 4:
        ks *= 2
    scaled = gscale is not None
    (y,) = _kernel("mmv")(
        inputs=[x, w, sc if sc is not None else dummy, bi if bi is not None else dummy,
                gscale.reshape(1) if scaled else _CONST["mmv_one"]],
        template=[("OT", out_dtype if ks == 1 else mx.float32), ("RT", x.dtype if scaled else mx.float32),
                  ("FMT", MMV_FMT[fmt]), ("I", I), ("O", O), ("SG", sg), ("RB", rb), ("KS", ks), ("NT", n)],
        grid=(tgs * 32 * sg, ks, 1), threadgroup=(32 * sg, 1, 1),
        output_shapes=[(ks * n, O)], output_dtypes=[out_dtype if ks == 1 else mx.float32])
    if ks > 1:
        y = y.reshape(ks, n, O).sum(axis=0)
        if scaled:
            y = y.astype(x.dtype).astype(mx.float32) * gscale
        y = y.astype(out_dtype)
    return y


def _mmv_rows(x):
    n = 1
    for d in x.shape[:-1]:
        n *= d
    return n

# A token-expert pair count at or below this uses the IQ2_XXS mat-vec kernel;
# larger batches (prefill) use the IQ2_XXS gather-QMM kernel.
SPEC_MAX = 8  # longest verify block (tokens) with rollback support
DECODE_PAIRS = 80  # 8 verify tokens x 9 experts use the decode kernel


_CONST = {}


def _iota(n, div=1):
    """mx.arange(n) // div as uint32, built once (no kernel per call)."""
    key = ("iota", n, div)
    if key not in _CONST:
        a = mx.arange(n, dtype=mx.uint32) // div
        mx.eval(a)
        _CONST[key] = a
    return _CONST[key]


def _zeros_u32(n):
    key = ("zeros", n)
    if key not in _CONST:
        a = mx.zeros((n,), mx.uint32)
        mx.eval(a)
        _CONST[key] = a
    return _CONST[key]


def _relu2(x):
    return mx.square(nn.relu(x))


class Q8Linear(nn.Module):
    """Q8_0 linear, transcoded at load time to MLX 8-bit affine (group 32)."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 4), mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // Q8_BLOCK), mx.float32)
        self.biases = mx.zeros((output_dims, input_dims // Q8_BLOCK), mx.float32)

    def __call__(self, x):
        n = _mmv_rows(x)
        if 1 < n <= MMV_MAX:
            O, I = self.weight.shape[0], self.weight.shape[1] * 4
            y = mmv(x.reshape(n, I), "q8", self.weight, O, I, self.scales, self.biases)
            return y.reshape(*x.shape[:-1], O)
        y = mx.quantized_matmul(
            x, self.weight, self.scales, self.biases,
            transpose=True, group_size=Q8_BLOCK, bits=8,
        )
        return y.astype(x.dtype)


class NVFP4Linear(nn.Module):
    """ModelOpt NVFP4 linear on the original bytes, with its global scale."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 8), mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 16), mx.uint8)
        # NVIDIA's weight_scale_2. Renamed in sanitize: mlx-vlm rewraps any
        # layer that has a weight_scale_2 tensor.
        self.global_scale = mx.zeros((), mx.float32)

    def __call__(self, x):
        n = _mmv_rows(x)
        if 1 < n <= MMV_MAX:
            O, I = self.weight.shape[0], self.weight.shape[1] * 8
            y = mmv(x.reshape(n, I), "nvfp4", self.weight, O, I, self.scales, gscale=self.global_scale)
            return y.reshape(*x.shape[:-1], O)
        y = mx.quantized_matmul(
            x, self.weight, self.scales,
            transpose=True, group_size=16, bits=4, mode="nvfp4",
        )
        return (y.astype(mx.float32) * self.global_scale).astype(x.dtype)


class IQ2Linear(nn.Module):
    """IQ2_XXS linear (shared experts). weight/scales are the qs/d planes."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        nb = input_dims // QK_K
        self.weight = mx.zeros((1, output_dims, nb * 16), mx.uint32)
        self.scales = mx.zeros((1, output_dims, nb), mx.float16)

    def __call__(self, x, relu2=False):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        n = x.shape[0]
        zeros = _zeros_u32(n)
        if 1 < n <= MMV_MAX:
            O, I = self.scales.shape[1], self.scales.shape[2] * QK_K
            y = mmv(_relu2(x) if relu2 else x, "iq2", self.weight[0], O, I, self.scales[0])
        elif n <= DECODE_PAIRS:
            y = iq2xxs_gather_qmv(x, self.weight, self.scales, _iota(n), zeros, relu2=relu2)
        else:
            if relu2:
                x = _relu2(x)
            y = iq2xxs_gather_qmm(x, self.weight, self.scales, zeros)
        return y.reshape(*shape[:-1], -1)


class IQ2Experts(nn.Module):
    """Stacked IQ2_XXS experts; weight/scales are the qs/d planes."""

    def __init__(self, num_experts: int, input_dims: int, output_dims: int):
        super().__init__()
        nb = input_dims // QK_K
        self.weight = mx.zeros((num_experts, output_dims, nb * 16), mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, nb), mx.float16)


class IQ2SwitchMLP(nn.Module):
    """Routed experts: relu2(x @ fc1[e].T) @ fc2[e].T, IQ2_XXS weights."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.fc1 = IQ2Experts(num_experts, input_dims, hidden_dims)
        self.fc2 = IQ2Experts(num_experts, hidden_dims, input_dims)

    def __call__(self, x, indices):
        lead = indices.shape  # (..., k)
        k = lead[-1]
        x = x.reshape(-1, x.shape[-1])
        ei = indices.reshape(-1).astype(mx.uint32)
        P = ei.shape[0]
        f1, f2 = self.fc1, self.fc2
        if P <= DECODE_PAIRS:
            h = iq2xxs_gather_qmv(x, f1.weight, f1.scales, _iota(P, k), ei)
            y = iq2xxs_gather_qmv(h, f2.weight, f2.scales, _iota(P), ei, relu2=True)
        else:
            order = mx.argsort(ei)
            inv = mx.argsort(order)
            es = ei[order]
            xs = x[order // k]
            tiles = expert_tiles(es, f1.weight.shape[0])
            h = _relu2(iq2xxs_gather_qmm(xs, f1.weight, f1.scales, es, tiles))
            y = iq2xxs_gather_qmm(h, f2.weight, f2.scales, es, tiles)[inv]
        return y.reshape(*lead, -1)


# ---------------------------------------------------------------------------
# Architecture. The Nemotron-H and Nemotron Omni classes below are vendored
# from mlx-vlm (MIT, Copyright (c) 2025 Prince Canuma), at the merge of
# Blaizzy/mlx-vlm#2528, so this file does not depend on mlx-vlm internals that
# change between versions. Imported from mlx-vlm, unchanged across those
# versions: the RADIO vision tower, its config, the caches, masks and
# compute_dt, and the Nemotron Omni processor registration.
# ---------------------------------------------------------------------------



@dataclass
class TextConfig(BaseModelConfig):
    """Nemotron-H language config, including Puzzle per-layer block_configs."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: list[str] | None = None
    layers_block_type: list[str] | None = None
    block_configs: list[dict] | None = None
    head_dim: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None
    n_group: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    topk_group: int | None = None
    num_experts_per_tok: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    time_step_limit: tuple[float, float] | None = None
    time_step_min: float | None = None
    time_step_max: float | None = None
    num_nextn_predict_layers: int = 0

    _block_type_to_char = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        pattern = (
            params.get("block_configs")
            or params.get("hybrid_override_pattern")
            or params.get("layers_block_type")
        )
        if pattern is not None:
            params.setdefault("num_hidden_layers", len(pattern))
        return super().from_dict(params)

    def __post_init__(self):
        if self.block_configs:
            self.layers_block_type = [b["block_type"] for b in self.block_configs]
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.time_step_limit is None:
            self.time_step_limit = (0.0, float("inf"))
        else:
            # Transformers writes non-finite floats as {"__float__": "Infinity"}.
            self.time_step_limit = tuple(
                float(v["__float__"]) if isinstance(v, dict) else v
                for v in self.time_step_limit
            )
        if self.hybrid_override_pattern is None and self.layers_block_type is not None:
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.hybrid_override_pattern is not None:
            self.num_hidden_layers = len(self.hybrid_override_pattern)

    def for_layer(self, index: int) -> "TextConfig":
        if not self.block_configs:
            return self
        overrides = {
            k: v for k, v in self.block_configs[index].items()
            if k != "block_type" and v is not None
        }
        return replace(self, block_configs=None, **overrides)


def _swap_linear(parent, name, cls):
    lin = getattr(parent, name)
    out_dims, in_dims = lin.weight.shape
    setattr(parent, name, cls(in_dims, out_dims))


class _GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, group_size: int):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)
        self.group_size = group_size

    def __call__(self, x, gate=None):
        if gate is not None:
            x = nn.silu(gate) * x
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


class _Mamba2Mixer(nn.Module):
    """mlx-vlm's NemotronHMamba2Mixer: the reference path for masked or
    batched (per-sequence length) calls."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim,
            kernel_size=args.conv_kernel, padding=0, groups=self.conv_dim,
            bias=args.use_conv_bias,
        )
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.mamba_proj_bias)
        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)
        self.norm = _GatedRMSNorm(
            self.intermediate_size, eps=args.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias)

    def _split_projected_states(self, projected):
        # Checkpoints may pad the projection with two unused d_mlp branches.
        base_size = self.intermediate_size + self.conv_dim + self.num_heads
        extra = projected.shape[-1] - base_size
        if extra < 0 or extra % 2:
            raise ValueError(f"invalid Nemotron-H Mamba projection width {projected.shape[-1]}")
        gate_start = extra
        conv_start = gate_start + self.intermediate_size
        dt_start = conv_start + self.conv_dim
        return (projected[..., gate_start:conv_start], projected[..., conv_start:dt_start],
                projected[..., dt_start:])

    def _conv(self, conv_input, cache, mask):
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, padded.shape[1] - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded, positions, axis=1)
            else:
                cache[0] = padded[:, -n_keep:, :]
        else:
            padded = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
        return nn.silu(self.conv1d(padded))

    def _ssm(self, hidden_states, B, C, dt, cache, mask):
        from mlx_vlm.models.ssm import ssm_update

        batch, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(batch, seq_len, self.num_heads, self.head_dim)
        B = B.reshape(batch, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch, seq_len, self.n_groups, self.ssm_state_size)
        state = cache[1] if cache else None
        y, state = ssm_update(
            hidden_states, self.A_log, B, C, self.D.astype(hidden_states.dtype), dt,
            self.dt_bias, state, self.time_step_limit, mask,
        )
        if cache:
            cache[1] = state
        return y.reshape(batch, seq_len, self.intermediate_size)

    def __call__(self, hidden_states, mask, cache=None):
        gate, conv_input, dt = self._split_projected_states(self.in_proj(hidden_states))
        conv_output = self._conv(conv_input, cache, mask)
        x, B, C = mx.split(
            conv_output,
            [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size],
            axis=-1,
        )
        y = self._ssm(x, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        return self.out_proj(self.norm(y, gate))


class _Attention(nn.Module):
    """mlx-vlm's NemotronHAttention (no position embedding)."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim if args.head_dim is not None else args.hidden_size // args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5
        bias = args.attention_bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


@mx.compile
def _group_expert_select(gates, bias, top_k, n_group, topk_group, scale, norm_topk_prob):
    orig = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)
    inds = mx.stop_gradient(mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k])
    scores = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, scores * scale


class _MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x):
        return _group_expert_select(
            x @ self.weight.T, self.e_score_correction_bias, self.top_k, self.n_group,
            self.topk_group, self.routed_scaling_factor, self.norm_topk_prob,
        )


_DEBUG = bool(__import__("os").environ.get("NEMOTRON_BOLT_DEBUG"))
_seen = set()


def _debug_once(*key):
    if key not in _seen:
        _seen.add(key)
        print("nemotron_bolt:", *key, flush=True)


class BoltMamba(_Mamba2Mixer):
    """Mamba2 mixer on fused kernels: conv + SiLU, then the selective scan, both
    reading the in_proj output in place. Decode (one token) is one compiled
    step. Masks and per-sequence lengths (batching) use mlx-vlm's code."""

    @staticmethod
    def _can_step(cache):
        return (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and getattr(cache, "lengths", None) is None
        )

    def __call__(self, hidden_states, mask=None, cache=None):
        if _DEBUG and hidden_states.shape[1] == 1:
            _debug_once("mamba", type(cache).__name__, mask is not None,
                        cache is not None and getattr(cache, "lengths", None) is not None)
        if mask is not None or (cache is not None and getattr(cache, "lengths", None) is not None):
            return super().__call__(hidden_states, mask=mask, cache=cache)
        if hidden_states.shape[1] == 1 and self._can_step(cache):
            if not hasattr(self, "_step"):
                self._step = mx.compile(self._decode_step)
            y, cache[0], cache[1] = self._step(hidden_states, cache[0], cache[1])
            cache.advance(1)
            return y
        b, L, _ = hidden_states.shape
        conv_state = cache[0] if cache is not None else None
        ssm_state = cache[1] if cache is not None else None
        if conv_state is None:
            conv_state = mx.zeros((b, self.conv_kernel_size - 1, self.conv_dim), hidden_states.dtype)
        if ssm_state is None:
            ssm_state = mx.zeros((b, self.num_heads, self.head_dim, self.ssm_state_size), mx.float32)
        stash = [] if cache is not None and L <= SPEC_MAX else None
        pre = (conv_state, ssm_state)
        y, conv_state, ssm_state = self._core(self.in_proj(hidden_states), conv_state, ssm_state, stash)
        if cache is not None:
            cache[0], cache[1] = conv_state, ssm_state
            cache.advance(L)
            # Speculative verify: what spec_rollback needs to keep only a prefix.
            cache.bolt_spec = (*pre, *stash) if stash else None
        return y

    def spec_rollback(self, cache, n):
        """Keep the first n tokens of the last multi-token call (exact: same scan kernel)."""
        conv_pre, ssm_pre, conv_in, u, dt = cache.bolt_spec
        k = self.conv_kernel_size - 1
        cache[0] = mx.concatenate([conv_pre, conv_in[:, :n]], axis=1)[:, -k:]
        _, cache[1] = ssm_scan(u[:, :n], self.A_log, self.D.astype(u.dtype), dt[:, :n], ssm_pre,
                               self.num_heads, self.head_dim, self.n_groups)
        cache.bolt_spec = None

    def _decode_step(self, x, conv_state, ssm_state):
        return self._core(self.in_proj(x), conv_state, ssm_state)

    def _core(self, p, conv_state, ssm_state, stash=None):
        base = self.intermediate_size + self.conv_dim + self.num_heads
        gate_start = p.shape[-1] - base  # two unused d_mlp branches, if any
        conv_start = gate_start + self.intermediate_size
        dt_start = conv_start + self.conv_dim
        u, conv_state = mamba_conv(p, conv_start, conv_state, self.conv1d.weight,
                                   self.conv1d.bias, self.conv_dim)
        dt = compute_dt(p[..., dt_start:], self.dt_bias, self.time_step_limit)
        if stash is not None:
            stash += [p[..., conv_start:dt_start], u, dt]
        y, ssm_state = ssm_scan(
            u, self.A_log, self.D.astype(p.dtype), dt, ssm_state,
            self.num_heads, self.head_dim, self.n_groups,
            lpc=32 if p.shape[1] == 1 else None,  # 32: the stock decode kernel's sums
        )
        gate = p[..., gate_start:conv_start]
        return self.out_proj(self.norm(y, gate)), conv_state, ssm_state


class BoltAttention(_Attention):
    """Long-context decode uses flash_decode; everything else is mlx-vlm's."""

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        if 1 < L <= SPEC_MAX and B == 1 and isinstance(mask, str) and type(cache).__name__ == "KVCache":
            return self._verify(x, cache)
        if L != 1 or mask is not None or cache is None or type(cache).__name__ != "KVCache":
            return super().__call__(x, mask=mask, cache=cache)
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        keys, values = cache.update_and_fetch(k, v)
        if (
            isinstance(keys, mx.array)
            and keys.shape[2] >= FLASH_DECODE_MIN_KEYS
            and flash_decode_supported(q, keys)
        ):
            if _DEBUG:
                _debug_once("flash decode", type(cache).__module__)
            out = flash_decode(q, keys, values, self.scale)
        else:
            out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def _verify(self, x, cache):
        """L query rows, each attending to the keys up to itself, one row per call."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        keys, values = cache.update_and_fetch(k, v)
        N = keys.shape[2]
        outs = []
        for i in range(L):
            n = N - L + 1 + i
            qi, ki, vi = q[:, :, i : i + 1], keys[:, :, :n], values[:, :, :n]
            if n >= FLASH_DECODE_MIN_KEYS and flash_decode_supported(qi, ki):
                outs.append(flash_decode(qi, ki, vi, self.scale))
            else:
                outs.append(mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.scale))
        out = mx.concatenate(outs, axis=2)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class BoltMoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        latent = config.moe_latent_size
        self.gate = _MoEGate(config)
        self.fc1_latent_proj = Q8Linear(config.hidden_size, latent)
        self.fc2_latent_proj = Q8Linear(latent, config.hidden_size)
        self.switch_mlp = IQ2SwitchMLP(latent, config.moe_intermediate_size, config.n_routed_experts)
        self.shared_experts = BoltMLP(config.hidden_size, config.moe_shared_expert_intermediate_size)

    def __call__(self, x):
        if x.shape[1] == 1:  # decode: fuse the router and glue ops
            if not hasattr(self, "_step"):
                self._step = mx.compile(self._forward)
            return self._step(x)
        return self._forward(x)

    def _forward(self, x):
        g = self.gate
        if g.n_group == 1 and g.norm_topk_prob and g.top_k > 1:
            inds, scores = route(x.reshape(-1, x.shape[-1]), g.weight, g.e_score_correction_bias,
                                 g.top_k, g.routed_scaling_factor)
            inds = inds.reshape(*x.shape[:-1], -1)
            scores = scores.reshape(*x.shape[:-1], -1)
        else:
            inds, scores = g(x)
        y = self.switch_mlp(self.fc1_latent_proj(x), inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return self.fc2_latent_proj(y) + self.shared_experts(x)


class BoltMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up_proj = IQ2Linear(hidden, intermediate)
        self.down_proj = IQ2Linear(intermediate, hidden)

    def __call__(self, x):
        return self.down_proj(self.up_proj(x), relu2=True)


class BoltBlock(nn.Module):
    """Pre-norm residual block. Decode (one token) runs the norm, mixer and
    residual as one compiled step."""

    def __init__(self, args: TextConfig, block_type: str):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.block_type = block_type
        if block_type == "M":
            self.mixer = BoltMamba(args)
            _swap_linear(self.mixer, "in_proj", NVFP4Linear)
            _swap_linear(self.mixer, "out_proj", NVFP4Linear)
        elif block_type == "*":
            self.mixer = BoltAttention(args)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _swap_linear(self.mixer, name, Q8Linear)
        elif block_type == "E":
            self.mixer = BoltMoE(args)
        else:
            raise ValueError(f"Nemotron Bolt has no {block_type!r} blocks")

    def __call__(self, x, mask=None, cache=None):
        if x.shape[1] == 1 and mask is None:
            if self.block_type == "E":
                if not hasattr(self, "_step"):
                    self._step = mx.compile(self._moe_step)
                return self._step(x)
            if self.block_type == "M" and self.mixer._can_step(cache):
                if not hasattr(self, "_step"):
                    self._step = mx.compile(self._mamba_step)
                y, cache[0], cache[1] = self._step(x, cache[0], cache[1])
                cache.advance(1)
                return y
        h = self.norm(x)
        if self.block_type in "M*":
            return x + self.mixer(h, mask=mask, cache=cache)
        return x + self.mixer(h)

    def _moe_step(self, x):
        if x.shape[0] == 1 and moe_decode_supported(self.mixer):
            return moe_decode(self.mixer, x, self.norm(x))
        return x + self.mixer._forward(self.norm(x))

    def _mamba_step(self, x, conv_state, ssm_state):
        if x.shape[0] == 1 and self.mixer.ssm_state_size % 32 == 0 and self.mixer.head_dim % MAMBA_CFG["TGD"] == 0:
            return mamba_decode(self.mixer, x, self.norm(x), conv_state, ssm_state)
        y, conv_state, ssm_state = self.mixer._decode_step(self.norm(x), conv_state, ssm_state)
        return x + y, conv_state, ssm_state


class BoltBackbone(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        pattern = args.hybrid_override_pattern
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [BoltBlock(args.for_layer(i), b) for i, b in enumerate(pattern)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        # Cache slots (Mamba and attention layers only) used for the masks.
        self.fa_idx = self.ssm_idx = 0
        for b in pattern:
            if b == "*":
                break
            if b == "M":
                self.fa_idx += 1
        for b in pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, capture_layer_ids=None,
                 hidden_sink=None):
        h = inputs_embeds if inputs_embeds is not None else self.embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        has_attention = any(layer.block_type == "*" for layer in self.layers)
        has_mamba = any(layer.block_type == "M" for layer in self.layers)
        attn_mask = create_attention_mask(h, cache[self.fa_idx] if has_attention else None)
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx] if has_mamba else None)
        capture = set(capture_layer_ids or ())
        ci = 0
        for index, layer in enumerate(self.layers):
            c = None
            if layer.block_type in "M*":
                c = cache[ci]
                ci += 1
            h = layer(h, mask=attn_mask if layer.block_type == "*" else ssm_mask, cache=c)
            if hidden_sink is not None and index in capture:
                hidden_sink.append(h)
        return self.norm_f(h)


class LanguageModel(nn.Module):
    # oMLX speculative decoding (DFlash block drafter): one row per verify,
    # at most SPEC_MAX tokens.
    _omlx_drafter_single_row = True
    _omlx_mtp_max_verify_rows = SPEC_MAX

    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = BoltBackbone(args)
        self.lm_head = Q8Linear(args.hidden_size, args.vocab_size)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.block_type == "M" else KVCache()
                for l in self.layers if l.block_type in "M*"]

    def get_input_embeddings(self, input_ids):
        return self.backbone.embeddings(input_ids)

    def sanitize(self, weights):
        return weights  # Model.sanitize did the work

    @property
    def cast_predicate(self):
        return lambda key: "e_score_correction_bias" not in key and "A_log" not in key

    def __call__(self, inputs=None, *, inputs_embeds=None, cache=None, output_hidden_states=False,
                 capture_layer_ids=None, return_hidden=False, **kwargs):
        """Speculative decoding: capture_layer_ids returns the outputs of those
        layers (for the drafter) as hidden_states, in the order asked; with
        return_hidden the final hidden state follows them."""
        kwargs.pop("n_confirmed", None)  # rollback is after the fact
        kwargs.pop("mask", None)
        shared_kv = {} if kwargs.pop("return_shared_kv", False) else None
        if not capture_layer_ids and not return_hidden:
            drafter = getattr(self, "_omlx_drafter", None)
            scope = getattr(drafter, "scope_uids", None)
            if scope and inputs is not None and inputs.ndim == 2 and inputs.shape == (len(scope), 1):
                # A plain decode step while a block drafter is attached: keep
                # the committed token in the drafter's context.
                out = self(inputs, cache=cache, capture_layer_ids=list(drafter.target_layer_ids))
                drafter.observe(scope, out.hidden_states)
                return out
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(inputs)
            if not output_hidden_states:
                h = self.backbone(cache=cache, inputs_embeds=inputs_embeds)
                return _LMOutput(logits=self.lm_head(h))
            # Nemotron Omni convention: each layer's input, then the normed output.
            sink = []
            h = self.backbone(cache=cache, inputs_embeds=inputs_embeds,
                              capture_layer_ids=range(len(self.layers) - 1), hidden_sink=sink)
            return _LMOutput(logits=self.lm_head(h), hidden_states=[inputs_embeds, *sink, h])
        sink = []
        ids = sorted(set(capture_layer_ids or ()))
        h = self.backbone(inputs, cache=cache, inputs_embeds=inputs_embeds,
                          capture_layer_ids=ids, hidden_sink=sink)
        by_id = dict(zip(ids, sink))
        captured = [by_id[i] for i in (capture_layer_ids or ())]
        if return_hidden:
            captured.append(h)
        return _LMOutput(logits=self.lm_head(h), hidden_states=captured,
                         gdn_states=[] if return_hidden else None, shared_kv_states=shared_kv)

    def rollback_speculative_cache(self, cache, gdn_states, accepted, block):
        """oMLX: keep accepted + 1 positions of the block-token verify call."""
        if not isinstance(accepted, int):
            accepted = [int(a) for a in (accepted.tolist() if hasattr(accepted, "tolist") else accepted)]
            if len(set(accepted)) != 1:
                raise ValueError("Nemotron Bolt rolls back one row at a time")
            accepted = accepted[0]
        self.spec_rollback(cache, accepted + 1, block)

    def spec_rollback(self, cache, n, total):
        """After a call of `total` tokens (<= SPEC_MAX), keep only its first n."""
        if n >= total:
            return
        ci = 0
        for layer in self.backbone.layers:
            if layer.block_type not in "M*":
                continue
            c = cache[ci]
            ci += 1
            if layer.block_type == "M":
                layer.mixer.spec_rollback(c, n)
            else:
                c.trim(total - n)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    sound_config: dict | None = None  # Bolt has no audio encoder
    model_type: str = "nemotron_h_nano_omni"
    force_image_size: int | None = None
    downsample_ratio: float = 0.5
    template: str | None = None
    ps_version: str = "v1"
    image_tag_type: str = "internvl"
    projector_hidden_size: int = 4096
    vit_hidden_size: int = 1280
    vision_final_layernorm: bool | None = None
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    img_context_token_id: int | None = None
    video_context_token_id: int | None = None
    sound_context_token_id: int | None = None
    eos_token_id: Any = None
    image_token_index: int | None = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        text = params.pop("text_config", None) or params.pop("llm_config", {})
        vision = params.pop("vision_config", None) or {}
        if params.get("sound_config"):
            raise ValueError("This Nemotron Omni file supports Bolt, which has no audio encoder")
        kwargs = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        config = cls(
            text_config=text if isinstance(text, TextConfig) else TextConfig.from_dict(text),
            vision_config=vision if isinstance(vision, VisionConfig) else VisionConfig.from_dict(vision),
            **kwargs,
        )
        if config.image_token_index is None:
            config.image_token_index = config.img_context_token_id
        return config


class _SquaredReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x) ** 2


class VisionProjection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        scale = int(1 / config.downsample_ratio)
        in_features = config.vit_hidden_size * scale**2
        self.layers = [
            nn.RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, config.projector_hidden_size, bias=False),
            _SquaredReLU(),
            nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _masked_scatter(final, mask, source):
    shape = final.shape
    flat = mx.flatten(final)
    positions = mx.array(np.where(mx.flatten(mask))[0], mx.uint32)
    flat[positions] = mx.flatten(source)
    return flat.reshape(shape)


def _retention_mask(video_embeds, pruning_rate):
    """Keep the first tubelet and the least similar tokens of later ones."""
    frames, tokens, _ = video_embeds.shape
    normed = video_embeds / mx.maximum(mx.linalg.norm(video_embeds, axis=-1, keepdims=True), 1e-8)
    similarity = mx.sum(normed[1:] * normed[:-1], axis=-1)
    scores = mx.concatenate([mx.full((1, tokens), 255, similarity.dtype), 1 - similarity])
    keep = max(tokens, int(frames * tokens * (1 - pruning_rate)))
    indices = mx.argsort(-scores.reshape(-1))[:keep]
    mask = mx.zeros((frames * tokens,), dtype=mx.bool_)
    mask[indices] = True
    return mask


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_model = VisionModel(config.vision_config)
        self.mlp1 = VisionProjection(config)
        self.vision_final_layernorm = (
            nn.LayerNorm(config.vit_hidden_size, eps=getattr(config.vision_config, "layer_norm_eps", 1e-6))
            if config.vision_final_layernorm else None
        )
        self.img_context_token_id = config.img_context_token_id
        self.video_temporal_patch_dim = config.video_temporal_patch_size
        self.video_pruning_rate = config.video_pruning_rate

    @property
    def layers(self):
        return self.language_model.layers

    def _merge_features(self, embeds, input_ids, token_id, features, name):
        features = features.reshape(-1, embeds.shape[-1]).astype(embeds.dtype)
        token_mask = input_ids == token_id
        n = int(mx.sum(token_mask).item())
        if n != features.shape[0]:
            raise ValueError(f"{name} token count ({n}) does not match feature count ({features.shape[0]}).")
        return _masked_scatter(embeds, mx.broadcast_to(token_mask[..., None], embeds.shape), features)

    def pixel_shuffle(self, x, scale_factor=0.5):
        batch, width, height, channels = x.shape
        x = x.reshape(batch, width, int(height * scale_factor), int(channels / scale_factor))
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(batch, int(height * scale_factor), int(width * scale_factor),
                      int(channels / (scale_factor * scale_factor)))
        if self.config.ps_version != "v1":
            x = x.transpose(0, 2, 1, 3)
        return x

    @staticmethod
    def _ensure_4d_pixels(pixel_values):
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]
        return pixel_values.astype(mx.float32)

    def _project_vision_features(self, vit_embeds, height, width):
        if self.vision_final_layernorm is not None:
            vit_embeds = self.vision_final_layernorm(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], height, width, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.config.downsample_ratio)
        return self.mlp1(vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]))

    def _patch_grid(self, pixel_values):
        patch = self.vision_model.radio_model.model.patch_generator.patch_size
        return pixel_values.shape[2] // patch, pixel_values.shape[3] // patch

    def extract_feature(self, pixel_values):
        if isinstance(pixel_values, (list, tuple)):
            hidden = self.config.text_config.hidden_size
            return mx.concatenate([self.extract_feature(pv).reshape(-1, hidden) for pv in pixel_values])
        pixel_values = self._ensure_4d_pixels(pixel_values)
        return self._project_vision_features(
            self.vision_model(pixel_values).features, *self._patch_grid(pixel_values))

    def extract_video_feature(self, pixel_values_videos):
        if isinstance(pixel_values_videos, (list, tuple)):
            frames = mx.concatenate([self._ensure_4d_pixels(pv) for pv in pixel_values_videos])
        else:
            frames = self._ensure_4d_pixels(pixel_values_videos)
        t = self.video_temporal_patch_dim
        n, c, height, width = frames.shape
        if n % t:
            frames = mx.concatenate([frames, mx.broadcast_to(frames[-1:], (t - n % t, c, height, width))])
            n = frames.shape[0]
        x = frames.reshape(n // t, t * c, height, width)
        return self._project_vision_features(
            self.vision_model(x, use_video_embedder=True).features, *self._patch_grid(frames))

    def get_input_embeddings(self, input_ids=None, pixel_values=None, mask=None, **kwargs):
        embeds = self.language_model.get_input_embeddings(input_ids)
        if pixel_values is not None:
            embeds = self._merge_features(embeds, input_ids, self.img_context_token_id,
                                          self.extract_feature(pixel_values), "Image")
        videos = kwargs.get("pixel_values_videos")
        if videos is not None:
            if input_ids.shape[0] != 1 or pixel_values is not None:
                raise ValueError("Nemotron video needs batch size 1 and no images in the same request.")
            video_features = self.extract_video_feature(videos)
            embeds = self._merge_features(embeds, input_ids, self.img_context_token_id,
                                          video_features, "Video")
            rate = kwargs.get("video_pruning_rate", self.video_pruning_rate)
            if not 0 <= rate <= 1:
                raise ValueError("video_pruning_rate must be between 0 and 1.")
            if rate > 0:
                keep_video = np.array(_retention_mask(video_features, rate))
                positions = np.flatnonzero(np.array(input_ids[0] == self.img_context_token_id))
                keep = np.ones(input_ids.shape[1], dtype=bool)
                keep[positions] = keep_video
                embeds = embeds[:, mx.array(np.flatnonzero(keep))]
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        embeds = self.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
        return self.language_model(input_ids, inputs_embeds=embeds.inputs_embeds, cache=cache, **kwargs)

    def sanitize(self, weights):
        text = self.config.text_config
        out = {}
        for key, value in weights.items():
            if key in ("vision_model.summary_idxs", "vision_model.radio_model.summary_idxs"):
                continue
            if key.startswith("vision_projector.vision_final_layernorm."):
                key = key.removeprefix("vision_projector.")
            if key.startswith("mlp1."):
                for i in (0, 1, 3):
                    key = key.replace(f"mlp1.{i}.", f"mlp1.layers.{i}.")
            out[key] = value
        weights = out

        def pop(name):
            return weights.pop(name)

        for i, block in enumerate(text.hybrid_override_pattern):
            p = f"language_model.backbone.layers.{i}.mixer"
            if block == "M":
                k = f"{p}.conv1d.weight"
                if weights[k].shape[-1] != 1:
                    weights[k] = weights[k].moveaxis(2, 1)
                for proj in ("in_proj", "out_proj"):
                    weights[f"{p}.{proj}.weight"] = pop(f"{p}.{proj}.weight").view(mx.uint32)
                    weights[f"{p}.{proj}.scales"] = pop(f"{p}.{proj}.weight_scale").view(mx.uint8)
                    weights[f"{p}.{proj}.global_scale"] = pop(f"{p}.{proj}.weight_scale_2")
            elif block == "*":
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _put_q8(weights, f"{p}.{proj}")
            elif block == "E":
                for proj in ("fc1_latent_proj", "fc2_latent_proj"):
                    _put_q8(weights, f"{p}.{proj}")
                n = text.for_layer(i).n_routed_experts
                for src, dst in (("up_proj", "fc1"), ("down_proj", "fc2")):
                    blocks = mx.stack([pop(f"{p}.experts.{e}.{src}.weight") for e in range(n)])
                    _put_iq2(weights, f"{p}.switch_mlp.{dst}", blocks)
                for proj in ("up_proj", "down_proj"):
                    k = f"{p}.shared_experts.{proj}"
                    _put_iq2(weights, k, pop(f"{k}.weight")[None])
        _put_q8(weights, "language_model.lm_head")
        return weights


def _put_iq2(weights, prefix, blocks):
    weights[f"{prefix}.weight"], weights[f"{prefix}.scales"] = iq2xxs_planes(blocks)


def _put_q8(weights, prefix):
    w, s, b = q8_0_to_affine(weights.pop(f"{prefix}.weight"))
    weights[f"{prefix}.weight"] = w
    weights[f"{prefix}.scales"] = s
    weights[f"{prefix}.biases"] = b


__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig", "AudioConfig",
           "LanguageModel", "VisionModel"]


# ---------------------------------------------------------------------------
# oMLX: speculative decoding with NVIDIA's LiLiCorr drafter (a dflash/ folder
# next to this file). oMLX versions without native Nemotron Bolt support run
# DFlash drafters only for model types on their list and install their verify
# loop only for those; add Bolt to both. oMLX with native support, plain
# mlx-vlm, or no dflash/ folder: nothing to do.
# ---------------------------------------------------------------------------

def _install_omlx_hooks():
    import importlib.util
    import os

    if not os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dflash")):
        return
    try:
        if importlib.util.find_spec("omlx.patches.nemotron_bolt") is not None:
            return
        from omlx.utils import model_loading as _ml
    except Exception:
        return
    types = getattr(_ml, "_DFLASH_BATCHED_MODEL_TYPES", None)
    if isinstance(types, dict) and "nemotron_h_nano_omni" not in types.get("vlm", ()):
        types["vlm"] = tuple(types.get("vlm", ())) + ("nemotron_h_nano_omni",)
    try:
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        apply_mlx_lm_mtp_patch()
    except Exception:
        pass


_install_omlx_hooks()
