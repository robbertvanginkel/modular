# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Accelerator tests for common_layers.rotary_embedding."""

from __future__ import annotations

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
)
from max.experimental.tensor import TensorType
from max.nn.rotary_embedding import YarnScalingParams


def _transfers(rope: RotaryEmbedding) -> list[str]:
    """Traces `forward` on an accelerator input and returns its transfers."""
    x_type = TensorType(DType.float32, [1, "seqlen", 1, 2], Accelerator())
    staged = rope.trace(x_type)
    return [line for line in str(staged).splitlines() if "mo.transfer" in line]


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_forward_on_accelerator() -> None:
    rope = RotaryEmbedding(
        dim=2, n_heads=1, theta=1e4, max_seq_len=1, device=Accelerator()
    )

    # The table is built on the host, and forward copies all of it to the
    # input's device, so the copy can be hoisted.
    assert rope.freqs_cis.device == CPU()
    rows, cols = rope.freqs_cis.shape
    [transfer] = _transfers(rope)
    assert f'!mo.tensor<[{rows}, {cols}], f32> to <"gpu", 0>' in transfer


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_yarn_forward_on_accelerator() -> None:
    rope = YarnRotaryEmbedding(
        dim=2,
        n_heads=1,
        theta=1e4,
        max_seq_len=1,
        device=Accelerator(),
        scaling_params=YarnScalingParams(
            factor=1.0,
            beta_fast=1.0,
            beta_slow=1.0,
            original_max_position_embeddings=1,
            truncate=False,
        ),
    )

    assert rope.freqs_cis.device == CPU()
    rows, cols = rope.freqs_cis.shape
    [transfer] = _transfers(rope)
    assert f'!mo.tensor<[{rows}, {cols}], f32> to <"gpu", 0>' in transfer
