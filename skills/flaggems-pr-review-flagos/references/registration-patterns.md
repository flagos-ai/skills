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

# Registration Patterns for FlagGems Operators

Every new operator in FlagGems must be registered in exactly three places. A missing or incorrect entry in any of these is a blocking issue.

---

## 1. `src/flag_gems/ops/__init__.py`

This file re-exports every operator so that the framework can discover it.

### Rules

- Use **absolute imports** from the submodule: `from flag_gems.ops.<module> import <function>`
- Add the function name to `__all__` in **alphabetical order**
- The imported name must match the wrapper function defined in the op module (not the Triton kernel)

### Correct

```python
from flag_gems.ops.abs import abs
from flag_gems.ops.add import add
from flag_gems.ops.foo import foo          # <-- new op
from flag_gems.ops.foo import foo_out      # <-- out variant
from flag_gems.ops.gelu import gelu

__all__ = [
    "abs",
    "add",
    "foo",        # <-- alphabetical
    "foo_out",
    "gelu",
]
```

### Incorrect

```python
# WRONG: relative import
from .foo import foo

# WRONG: importing the triton kernel instead of the wrapper
from flag_gems.ops.foo import foo_kernel

# WRONG: not in alphabetical order in __all__
__all__ = [
    "abs",
    "gelu",
    "foo",   # <-- should come before "gelu"
]
```

---

## 2. `src/flag_gems/__init__.py` (`_FULL_CONFIG` list)

This list maps PyTorch op names to their FlagGems wrapper functions, telling the dispatch system which ops to override.

### Format

```python
("op_name", wrapper_function),
```

### Rules

- **NO `aten::` prefix** -- just the bare op name string
- **Overloads** use a dot separator: `"foo.Tensor"`, `"foo.out"`
- **Inplace** ops use trailing underscore: `"foo_"`
- **`special.*` ops** keep the dot namespace: `("special.erfc", special_erfc)` -- do NOT flatten to `"special_erfc"`
- The second element must be the **wrapper function**, not the Triton kernel
- Entries should be in **alphabetical order** by op name

### Correct

```python
_FULL_CONFIG = [
    ("abs", abs),
    ("add.Tensor", add),
    ("add.out", add_out),
    ("foo", foo),                        # <-- new op
    ("foo_", foo_),                      # <-- inplace variant
    ("foo.out", foo_out),                # <-- out variant
    ("special.erfc", special_erfc),      # <-- special namespace preserved
]
```

### Incorrect

```python
# WRONG: aten:: prefix
("aten::foo", foo),

# WRONG: special namespace flattened
("special_erfc", special_erfc),

# WRONG: mapping to kernel instead of wrapper
("foo", foo_kernel),

# WRONG: overload uses underscore instead of dot
("foo_Tensor", foo),
```

---

## 3. `operators.yaml`

The YAML registry describes metadata for each operator. It is consumed by CI, docs, and the benchmark/test discovery system.

### Required Fields

| Field | Description |
|-------|-------------|
| `id` | Operator identifier (alphabetical insertion, no leading underscore) |
| `description` | One-line description from PyTorch docs |
| `for` | List of aten op names that this entry covers (must match `_FULL_CONFIG` entries) |
| `labels` | Must include both `"aten"` and `"KernelGen"` |
| `kind` | One of: `Math`, `Reduction`, `NeuralNetwork`, `LinearAlg`, `Tensor`, `Logic` |
| `stages` | Release stage info, e.g. `alpha: '5.1'` |

### Correct

```yaml
- id: foo
  description: Computes the foo of each element in input.
  for:
    - foo
    - foo.out
    - foo_
  labels:
    - aten
    - KernelGen
  kind: Math
  stages:
    alpha: '5.1'
```

### Incorrect

```yaml
# WRONG: id has leading underscore
- id: _foo
  description: Computes the foo of each element in input.
  for:
    - foo
  labels:
    - aten
    - KernelGen
  kind: Math
  stages:
    alpha: '5.1'

# WRONG: missing "KernelGen" label
- id: foo
  description: Computes foo.
  for:
    - foo
  labels:
    - aten
  kind: Math
  stages:
    alpha: '5.1'

# WRONG: "for" entry doesn't match _FULL_CONFIG
- id: foo
  description: Computes foo.
  for:
    - aten::foo          # <-- no aten:: prefix here either
  labels:
    - aten
    - KernelGen
  kind: Math
  stages:
    alpha: '5.1'

# WRONG: kind is not a valid category
- id: foo
  description: Computes foo.
  for:
    - foo
  labels:
    - aten
    - KernelGen
  kind: Pointwise         # <-- not valid; use "Math" for pointwise ops
  stages:
    alpha: '5.1'

# WRONG: stages uses number instead of quoted string
- id: foo
  description: Computes foo.
  for:
    - foo
  labels:
    - aten
    - KernelGen
  kind: Math
  stages:
    alpha: 5.1            # <-- must be quoted: '5.1'
```

---

## Reviewer Checklist

1. All three files updated in the same PR?
2. Names are consistent across all three registration points?
3. Alphabetical ordering maintained in each file?
4. Wrapper function (not kernel) referenced in both `ops/__init__.py` and `_FULL_CONFIG`?
5. `for` entries in YAML exactly match the string keys in `_FULL_CONFIG`?
6. Labels include both `"aten"` and `"KernelGen"`?
7. `stages: alpha: '5.1'` present for new ops?
