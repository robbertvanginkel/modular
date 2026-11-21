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
"""Tests for max.nn.Linear."""

from __future__ import annotations

import pytest
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.tensor import Tensor


def test_repr() -> None:
    assert repr(Linear(2, 2)) == "Linear(in_dim=Dim(2), out_dim=Dim(2))"
    assert (
        repr(Linear(1, 3, bias=False))
        == "Linear(in_dim=Dim(1), out_dim=Dim(3), bias=False)"
    )


def test_in_dim() -> None:
    assert Linear(2, 3).in_dim == 2


def test_out_dim() -> None:
    assert Linear(2, 3).out_dim == 3


def test_parameters() -> None:
    linear = Linear(2, 3)
    assert dict(linear.parameters) == {
        "weight": linear.weight,
        "bias": linear.bias,
    }


def test_call() -> None:
    linear = Linear(2, 3)
    result = linear(Tensor.ones([2]))
    assert result.shape == [3]


def test_repr_accum_dtype() -> None:
    assert (
        repr(Linear(2, 2, accum_dtype=DType.float32))
        == "Linear(in_dim=Dim(2), out_dim=Dim(2), accum_dtype=float32)"
    )


def test_accum_dtype_with_quant_config_raises() -> None:
    quant_config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.TENSOR, dtype=DType.float32
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )
    with pytest.raises(ValueError, match="accum_dtype"):
        Linear(2, 2, quant_config=quant_config, accum_dtype=DType.float32)


# `x @ w.T` is 2 + 2**-8 + 2**-10, which rounds to 2.0 in bfloat16. Adding the
# bias -1 gives exactly 1 + 2**-8 + 2**-10, which rounds to 1 + 2**-7. Rounding
# after the matmul and again after the bias gives 1.0 instead.
_X = [[2.0, 2.0**-8, 2.0**-10]]
_W = [[1.0, 1.0, 1.0]]
_B = [-1.0]


@pytest.mark.parametrize(
    "accum_dtype, expected",
    [(None, 1.0), (DType.float32, 1.0 + 2.0**-7)],
    ids=["default", "float32"],
)
def test_bfloat16_bias_rounding(
    accum_dtype: DType | None, expected: float
) -> None:
    linear = Linear(3, 1, accum_dtype=accum_dtype)
    linear.weight = Tensor(_W, dtype=DType.bfloat16, device=CPU())
    linear.bias = Tensor(_B, dtype=DType.bfloat16, device=CPU())
    result = linear(Tensor(_X, dtype=DType.bfloat16, device=CPU()))
    assert result.dtype == DType.bfloat16
    assert result.cast(DType.float32).item() == expected


def test_accum_dtype_without_bias() -> None:
    linear = Linear(3, 1, bias=False, accum_dtype=DType.float32)
    linear.weight = Tensor(_W, dtype=DType.bfloat16, device=CPU())
    result = linear(Tensor(_X, dtype=DType.bfloat16, device=CPU()))
    assert result.dtype == DType.bfloat16
    # 2 + 2**-8 + 2**-10 rounds to 2.0 in bfloat16.
    assert result.cast(DType.float32).item() == 2.0
