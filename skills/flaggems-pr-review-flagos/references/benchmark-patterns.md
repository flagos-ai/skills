<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Benchmark Patterns for FlagGems Operators

Benchmark files live in `benchmark/` and follow strict conventions for class selection, dtype usage, and naming.

---

## Benchmark Class Selection

| Operator Type | Class | Example Ops |
|---|---|---|
| Unary pointwise (one input, element-wise) | `base.UnaryPointwiseBenchmark` | sin, cos, abs, exp, neg, rsqrt |
| Binary pointwise (two inputs, element-wise) | `base.BinaryPointwiseBenchmark` | add, mul, sub, div, pow |
| Reduction (collapses a dimension) | `base.UnaryReductionBenchmark` | sum, mean, max, min, prod, any |
| BLAS (matrix multiply) | `base.BlasBenchmark` | mm, bmm, addmm, matmul |
| Custom input generation | `base.GenericBenchmark` with `input_fn` | dropout, where, scatter |
| Custom shapes | Inherit `GenericBenchmark`, override `set_more_shapes` | conv2d, embedding |

---

## dtype Rule

- All benchmarks **must** use `consts.FLOAT_DTYPES` for the dtype parametrization
- Exception: ops that genuinely do not support fp16/bf16 (e.g., some integer-only ops) may hardcode dtypes **but must include a comment explaining why**

### Correct

```python
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
```

### Incorrect

```python
# WRONG: hardcoded without explanation
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])

# ACCEPTABLE: hardcoded WITH explanation
# lgamma does not support fp16/bf16 on most hardware
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
```

---

## What to Flag in Review

- Custom `pytest.parametrize` for shapes/dtypes instead of using the benchmark class infrastructure
- Wrong benchmark class for the operator type (e.g., using `GenericBenchmark` for a simple unary op)
- Hardcoded dtype list without a comment explaining the restriction
- `@pytest.mark.<op_name>` does not match the `id` field in `operators.yaml`
- Missing benchmark file entirely for a new operator

---

## Correct Templates

### Unary Pointwise (e.g., sin)

```python
import pytest
import torch

from .. import base
from .. import consts


@pytest.mark.sin
def test_perf_sin():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sin",
        torch_op=torch.sin,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### Binary Pointwise (e.g., add)

```python
import pytest
import torch

from .. import base
from .. import consts


@pytest.mark.add
def test_perf_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### Reduction (e.g., sum)

```python
import pytest
import torch

from .. import base
from .. import consts


@pytest.mark.sum
def test_perf_sum():
    bench = base.UnaryReductionBenchmark(
        op_name="sum",
        torch_op=torch.sum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### BLAS (e.g., mm)

```python
import pytest
import torch

from .. import base
from .. import consts


@pytest.mark.mm
def test_perf_mm():
    bench = base.BlasBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### Generic with Custom Input (e.g., where)

```python
import pytest
import torch

from .. import base
from .. import consts


@pytest.mark.where
def test_perf_where():
    def input_fn(shape, dtype, device):
        cond = torch.randn(shape, device=device) > 0
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        return cond, x, y

    bench = base.GenericBenchmark(
        op_name="where",
        torch_op=torch.where,
        input_fn=input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### Generic with Custom Shapes (e.g., embedding)

```python
import pytest
import torch

from .. import base
from .. import consts


class EmbeddingBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Embedding uses (num_embeddings, embedding_dim) + index shapes
        self.shapes = [
            (1024, 256),
            (4096, 512),
            (32000, 768),
        ]


@pytest.mark.embedding
def test_perf_embedding():
    def input_fn(shape, dtype, device):
        num_embeddings, embedding_dim = shape
        weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(0, num_embeddings, (32, 128), device=device)
        return weight, indices

    bench = EmbeddingBenchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        input_fn=input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

---

## Reviewer Checklist

1. Correct benchmark class for the operator type?
2. Uses `consts.FLOAT_DTYPES` (or has a comment if hardcoded)?
3. `@pytest.mark.<name>` matches `operators.yaml` id?
4. `op_name` string matches `operators.yaml` id?
5. `torch_op` points to the correct PyTorch reference function?
6. Custom `input_fn` generates tensors on the correct device and with the correct dtype?
7. No stray `print()` or debug statements?

---

## Benchmark Fairness

- The torch reference and gems operation must measure the same computation scope. If one side includes extra work (e.g., forward pass), the comparison is invalid and speedup numbers are misleading.
- **Common mistake:** A benchmark for a backward op wraps torch's forward+backward together but the gems side only runs the backward kernel. This inflates the speedup ratio because the torch measurement includes extra forward computation.
- **Correct pattern for backward benchmarks:** Use `torch.ops.aten.<op>_backward(...)` directly as the torch reference, so both sides measure only the backward computation.
- **No duplicate shapes:** Custom benchmark shapes defined via `set_more_shapes` should not duplicate entries already present in `core_shapes.yaml`. Check for overlap before adding custom shapes.
