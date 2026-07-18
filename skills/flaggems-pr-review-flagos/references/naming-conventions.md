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

# FlagGems Naming Conventions

Complete naming convention reference for reviewers. Use this to verify consistency across kernel files, tests, benchmarks, yaml config, and registration.

---

## 1. Master Naming Table

Given an operator named `foo`, `_foo` (leading underscore), or `foo_` (inplace/trailing underscore):

| Artifact | `foo` | `_foo` | `foo_` (inplace) | `foo_out` (out variant) |
|----------|-------|--------|-------------------|------------------------|
| **Kernel file** | `foo.py` | `_foo.py` | `foo_.py` | `foo_out.py` |
| **Test file** | `test_foo.py` | `test_foo.py` | `test_foo_.py` | `test_foo_out.py` |
| **Benchmark file** | `test_foo_perf.py` | `test_foo_perf.py` | `test_foo__perf.py` | `test_foo_out_perf.py` |
| **yaml id** | `foo` | `foo` | `foo_` | `foo_out` |
| **pytest mark** | `@pytest.mark.foo` | `@pytest.mark.foo` | `@pytest.mark.foo_` | `@pytest.mark.foo_out` |
| **benchmark op_name** | `"foo"` | `"foo"` | `"foo_"` | `"foo_out"` |
| **`__init__.py` import** | `from flag_gems.ops.foo import foo` | `from flag_gems.ops._foo import _foo` | `from flag_gems.ops.foo_ import foo_` | `from flag_gems.ops.foo_out import foo_out` |
| **`__all__` entry** | `"foo"` | `"_foo"` | `"foo_"` | `"foo_out"` |
| **`_FULL_CONFIG` entry** | `"foo"` | `"_foo"` | `"foo_"` | `"foo_out"` |

---

## 2. Key Rules

### Leading underscore (`_foo`)

- **Strip** the leading underscore in: yaml id, pytest mark, benchmark op_name, test file name, benchmark file name.
- **Keep** the leading underscore in: kernel file name (`_foo.py`), `__init__.py` import path and function name, `__all__` entry.

**Rationale:** The leading underscore indicates a Python-private helper. The yaml/mark/op_name reference the ATen operator name (which has no leading underscore). File-system and import paths preserve the actual module name.

### Trailing underscore (`foo_`, inplace)

- **Keep** the trailing underscore everywhere. It is part of the operator identity.
- Note: benchmark file for `foo_` becomes `test_foo__perf.py` (double underscore before `perf` is acceptable here because one underscore belongs to the op name and one is the separator).

### Overload variants (e.g., `eq` vs `eq_scalar`, `norm` vs `norm.ScalarOpt_dim`)

- Each overload gets its **own** yaml id, pytest mark, and benchmark op_name.
- Example:

| | `eq` (Tensor) | `eq_scalar` (Scalar) |
|--|--|--|
| yaml id | `eq` | `eq_scalar` |
| pytest mark | `@pytest.mark.eq` | `@pytest.mark.eq_scalar` |
| op_name | `"eq"` | `"eq_scalar"` |
| kernel file | `eq.py` | `eq_scalar.py` |
| test file | `test_eq.py` | `test_eq_scalar.py` |

- Overloads MUST NOT share a single mark. Each variant must be independently selectable via `pytest -m`.

### `_FULL_CONFIG` entries

- Entries in `_FULL_CONFIG` use the **bare operator name** without an `aten::` prefix.
- Correct: `"foo"`, `"foo_"`, `"foo_out"`
- Wrong: `"aten::foo"`, `"aten::foo_"`

### Test file naming

- Test files always strip the leading underscore: `test_foo.py`, never `test__foo.py`.
- Benchmark files also strip the leading underscore: `test_foo_perf.py`, never `test__foo_perf.py`.

---

## 3. Reviewer Checklist: What to Flag

| Issue | Example of violation | Correct form |
|-------|---------------------|--------------|
| Mark doesn't match yaml id | yaml: `abs`, mark: `@pytest.mark.fabs` | Mark must be `@pytest.mark.abs` |
| op_name doesn't match yaml id | yaml: `cross_entropy`, op_name: `"CrossEntropy"` | op_name must be `"cross_entropy"` |
| Double underscore in test filename (leading _) | `test__foo.py` for op `_foo` | `test_foo.py` |
| `_FULL_CONFIG` has aten:: prefix | `"aten::softmax"` | `"softmax"` |
| Overload variants sharing a single mark | Both `ge` and `ge_scalar` use `@pytest.mark.ge` | `ge_scalar` needs `@pytest.mark.ge_scalar` |
| Leading underscore NOT stripped in yaml id | yaml id: `_foo` | yaml id: `foo` |
| Leading underscore stripped in import | `from flag_gems.ops.foo import foo` for op `_foo` | `from flag_gems.ops._foo import _foo` |
| Trailing underscore lost in mark | `@pytest.mark.add` for inplace `add_` | `@pytest.mark.add_` |
| Mismatched kernel filename | Op `_foo` in file `foo.py` | Should be `_foo.py` |

---

## 4. Quick Decision Flowchart

```
Is there a leading underscore in the Python function name?
  YES -> Strip it for: yaml id, mark, op_name, test filename, benchmark filename
         Keep it for: kernel filename, import path, __all__, _FULL_CONFIG entry
  NO  -> Use the name as-is everywhere

Is this an overload variant (e.g., Scalar vs Tensor signature)?
  YES -> Give it a distinct yaml id, mark, and op_name (usually suffix: _scalar, _tensor, etc.)
  NO  -> Single yaml id / mark / op_name

Does the name end with underscore (inplace)?
  YES -> Keep trailing underscore in ALL artifacts
  NO  -> Standard naming
```
