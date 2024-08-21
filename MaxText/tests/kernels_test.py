"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for kernels """

import jax.numpy as jnp
import numpy as np

import pytest
import unittest

from absl import flags
from jax import random
from kernels.ragged_attention import ragged_mha, reference_mha, ragged_mqa, reference_mqa


class RaggedAttentionTest(unittest.TestCase):
  """Tests for ragged attention kernel."""
  batch_size = 4
  num_kv_heads = 32
  num_query_heads = 32
  max_prefill_predict_length = 256
  max_target_length = 512
  head_dim = 128

  seed = 2
  dtype = jnp.float32
  key = random.key(seed)


  @pytest.mark.tpu
  def test_ragged_mqa(self):
    k1, k2, k3 = random.split(self.key, 3)
    q = random.normal(k1, (self.batch_size, 1, self.head_dim), dtype=self.dtype)
    k = random.normal(k2, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    v = random.normal(k3, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, ragged_max, ragged_denom = ragged_mqa(q, k, v, lengths)
    reference_out, reference_max, reference_denom = reference_mqa(q, k, v, lengths)
    assert jnp.allclose(ragged_out, reference_out, rtol=0.0, atol=1e-1), f"Max difference {jnp.max(abs(ragged_out - reference_out))} > 1e-1"


  @pytest.mark.tpu
  def test_ragged_mha(self):
    k1, k2, k3 = random.split(self.key, 3)
    q = random.normal(k1, (self.batch_size, 1, self.num_query_heads, self.head_dim), dtype=self.dtype)
    k = random.normal(k2, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    v = random.normal(k3, (self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    lengths = jnp.array(np.random.randint(1, self.max_target_length, self.batch_size), dtype=jnp.int32)

    ragged_out, ragged_max, ragged_denom = ragged_mha(q, k, v, lengths)
    reference_out, reference_max, reference_denom = reference_mha(q, k, v, lengths)
    ragged_out = ragged_out / ragged_denom
    assert jnp.allclose(ragged_out, reference_out, rtol=0.0, atol=1e-1), f"Max difference {jnp.max(abs(ragged_out - reference_out))} > 1e-1"


if __name__ == "__main__":
  unittest.main()