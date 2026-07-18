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

# Test Patterns for FlagGems Operators

Test files live in `tests/` and validate numerical accuracy against PyTorch reference implementations.

---

## Required Elements

| Element | Requirement |
|---|---|
| File name | `tests/test_<op_id>.py` where `op_id` is the yaml id (no leading underscore) |
| Import | `from . import accuracy_utils as utils` (relative import) |
| Mark | `@pytest.mark.<op_id>` matching the `id` field in `operators.yaml` |
| Reference | `utils.to_reference(inp)` called on ALL input tensors before computing reference |
| Assertion (float) | `utils.gems_assert_close(res, ref, dtype)` |
| Assertion (bitwise/exact) | `utils.gems_assert_equal(res, ref)` |
| Parametrize dtypes | `utils.FLOAT_DTYPES` |
| Parametrize shapes | `utils.POINTWISE_SHAPES` (or appropriate shape constant) |

---

## Correct Template

```python
import pytest
import torch

from . import accuracy_utils as utils


@pytest.mark.foo
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_foo(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    # Reference: must use to_reference on ALL inputs
    ref_inp = utils.to_reference(inp)
    ref = torch.foo(ref_inp)

    # Result: run on FlagGems
    res = torch.foo(inp)

    # Assert
    utils.gems_assert_close(res, ref, dtype)
```

### Out-variant Template

```python
@pytest.mark.foo_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_foo_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp)
    ref = torch.foo(ref_inp)

    torch.foo(inp, out=out)

    utils.gems_assert_close(out, ref, dtype)
```

### Inplace-variant Template

```python
@pytest.mark.foo_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_foo_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_inp = utils.to_reference(inp.clone())
    ref_inp.foo_()

    inp.foo_()

    utils.gems_assert_close(inp, ref_inp, dtype)
```

### NaN-producing Op Template (e.g., log, sqrt on negative inputs)

```python
@pytest.mark.log
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_log(shape, dtype):
    # Input may contain negatives -> NaN in output
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_inp = utils.to_reference(inp)
    ref = torch.log(ref_inp)

    res = torch.log(inp)

    utils.gems_assert_close(res, ref, dtype, equal_nan=True)
```

---

## What to Flag in Review

### Blocking Issues

| Issue | Why It Matters |
|---|---|
| Absolute import: `from flag_gems.testing import ...` | Must use relative: `from . import accuracy_utils as utils` |
| Missing `to_reference` on input tensors | Reference computation runs on wrong device/precision; test is meaningless |
| `rtol` parameter in `gems_assert_close` | Framework manages tolerances internally; manual rtol masks real failures |
| `print()` anywhere in test | CI noise; must be removed |
| Wrong mark (doesn't match yaml id) | Test won't be discovered by the mark-based runner |
| `test_accuracy_` or `test_perf_` prefix | Function names should be `test_accuracy_<op>` for accuracy tests only; `test_perf_` belongs in benchmark files |
| NaN-producing op without `equal_nan=True` | Test will spuriously fail on valid NaN outputs |

### Common Anti-patterns

```python
# WRONG: absolute import
from flag_gems.testing import accuracy_utils as utils

# WRONG: missing to_reference -- comparing GPU result against GPU reference
ref = torch.foo(inp)  # should be torch.foo(utils.to_reference(inp))

# WRONG: custom rtol -- hides real precision issues
utils.gems_assert_close(res, ref, dtype, rtol=1e-3)

# WRONG: print left in test
print(f"res={res}, ref={ref}")
utils.gems_assert_close(res, ref, dtype)

# WRONG: mark doesn't match yaml id
@pytest.mark.floor_divide   # yaml id is "floor_div"
def test_accuracy_floor_div(shape, dtype):
    ...

# WRONG: NaN op without equal_nan
def test_accuracy_log(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = utils.to_reference(inp)
    ref = torch.log(ref_inp)
    res = torch.log(inp)
    utils.gems_assert_close(res, ref, dtype)  # <-- will fail on NaN entries
```

---

## Reviewer Checklist

1. File named `test_<op_id>.py` with `op_id` matching `operators.yaml`?
2. Uses relative import `from . import accuracy_utils as utils`?
3. `@pytest.mark.<op_id>` present and correct?
4. ALL input tensors wrapped with `utils.to_reference()` before computing reference?
5. Uses `gems_assert_close` (float) or `gems_assert_equal` (exact) -- no raw `torch.allclose`?
6. No manual `rtol` parameter? (`atol` is allowed)
7. No `print()` statements?
8. NaN-producing ops use `equal_nan=True`?
9. Parametrized with `utils.FLOAT_DTYPES` and appropriate shape constants?
10. Test function name follows `test_accuracy_<op_variant>` convention?
