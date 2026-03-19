---
name: kernelgen-for-flaggems
description: >
  Generate a FlagGems operator via kernelgen-mcp. Checks environment & MCP availability, calls the
  code generator, places files in the correct FlagGems project locations, and runs accuracy +
  benchmark tests. Use this skill when working in a FlagGems repository and need to generate GPU
  kernel operators. Trigger when the user says things like "generate a FlagGems operator", "create
  a kernel for FlagGems", or "/kernelgen-for-flaggems".
argument-hint: "<operator_name> [--func-type <type>]"
user-invokable: true
compatibility: "Python 3.8+, FlagGems, PyTorch with CUDA, Triton"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: gpu-kernel-generation
  tags: [kernelgen, flaggems, triton, gpu, mcp, operator-generation]
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

# KernelGen Skill — Generate FlagGems Operators via MCP

You are an expert at generating GPU kernel operators for the FlagGems project using the `kernelgen-mcp` MCP service.

**Tool usage**: This skill relies on the following capabilities:
- **Shell**: execute shell commands (python, pip, pytest, etc.) via the Bash/shell tool
- **Read**: read files from the codebase
- **Write / Edit**: create or modify files
- **Grep**: search file contents
- **Glob**: find files by pattern
- **MCP tools**: `mcp__kernelgen-mcp__generate_operator` and `mcp__kernelgen-mcp__optimize_triton`

> **⚠️ MCP Prerequisite Check**: If the user has not configured the kernelgen MCP service (i.e., MCP tools are unavailable or calls fail),
> immediately prompt the user to visit https://kernelgen.flagos.io/ to register and obtain the kernelgen MCP service URL and JWT Token,
> then complete the configuration following Step 0b instructions before retrying. Do not proceed with subsequent steps if MCP is not ready.

When you need user input or clarification, ask a question directly and wait for their reply.
Always use the appropriate built-in tool rather than outputting commands for the user to run manually.

## Step 0: Pre-flight — Environment & MCP Check

### 0a. Check Python Environment

Run the following diagnostic command:

```bash
python -c "import torch; import triton; import flag_gems; print('torch', torch.__version__); print('triton', triton.__version__); print('flag_gems', flag_gems.__version__); print('device', flag_gems.device)"
```

**If any import fails**, identify what's missing and use the shell tool to install it:

| Missing package | Install command |
|---|---|
| `torch` | Do NOT auto-install. Ask the user for the correct install command, as the CUDA wheel variant depends on their environment. |
| `triton` | `pip install triton` |
| `flag_gems` | `pip install -e .` from the repo root, or `pip install flag_gems` |
| `pytest` | `pip install pytest` |
| `numpy` | `pip install numpy` |
| `scipy` | `pip install scipy` |
| `pyyaml` | `pip install pyyaml` |
| `packaging` | `pip install packaging` |

If `torch` is missing or has no CUDA support, run a separate GPU diagnostic:

```bash
python -c "import torch; print('cuda_version:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available())"
```

Then diagnose:
- If `torch.version.cuda` is `None` → torch is a CPU-only build. Ask the user to reinstall with CUDA.
- If `torch.version.cuda` has a value but `torch.cuda.is_available() == False` → CUDA build but no GPU
  detected. Warn the user that GPU tests will fail and ask how to proceed.

If `flag_gems` is not installed, first verify `pyproject.toml` or `setup.py` exists in the current
working directory (to confirm we're at the repo root). If found, run `pip install -e .`. If not
found, ask the user for the correct repo root path.

After installing, re-run the diagnostic via the shell tool to confirm everything works. Only proceed when all imports succeed.

### 0b. Check MCP Availability

Verify that the `kernelgen-mcp` MCP server is configured and reachable.

Use the Read tool to read `.claude/settings.json` and look for an `mcpServers` entry whose key
contains `kernelgen` (case-insensitive). If the file does not exist, treat it as "MCP not configured".

**If the MCP server is NOT configured**, stop and print the following message to the user, then
wait for the user to provide the URL and JWT token:

```
The kernelgen-mcp service is not yet configured. Please follow these steps:

1. Visit https://kernelgen.flagos.io/ to register and obtain a JWT Token.
2. Add the following MCP configuration to `.claude/settings.json` (replace <YOUR_URL> and <YOUR_JWT_TOKEN> with actual values):

{
  "mcpServers": {
    "kernelgen-mcp": {
      "type": "sse",
      "url": "<YOUR_URL>",
      "headers": {
        "Authorization": "Bearer <YOUR_JWT_TOKEN>"
      }
    }
  }
}

3. After configuration is complete, please re-run this command.
```

After the user provides the URL and JWT, use the Edit tool (or Write tool if the file doesn't exist)
to write the configuration into `.claude/settings.json`, merging with any existing content. Then proceed.

**If the MCP server IS configured**, proceed to Step 1.

## Step 1: Understand the Operator Request

Parse the user's description to determine:
- `kernel_name`: operator name (e.g., `relu`, `softmax`, `gelu`, `layer_norm`).
  Prefer names exactly as defined in `torch.ops.aten` (e.g., `relu`, `silu`, `neg`),
  since registration uses the aten name.
- `torch_call`: the actual Python call to invoke this op on a tensor. This is NOT always
  `torch.<kernel_name>`. Determine it as follows:
  1. If `torch.<kernel_name>` exists (`hasattr(torch, '<kernel_name>')` is True) → use `torch.<kernel_name>`
  2. If not, check `torch.nn.functional.<kernel_name>` (e.g., `torch.nn.functional.layer_norm`)
  3. If not, check `torch.special.<kernel_name>` or `torch.linalg.<kernel_name>`
  4. As a last resort, use `torch.ops.aten.<kernel_name>.default`
  Do NOT use `torch._C._nn` — it is an internal API that can break across PyTorch versions.
  **Mandatory extra arguments**: If the chosen `torch_call` requires mandatory arguments beyond the
  input tensor(s), you MUST include them with sensible defaults. Common cases:
  - `torch.nn.functional.softmax(x, dim=-1)`
  - `torch.nn.functional.log_softmax(x, dim=-1)`
  - `torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])`
  - `torch.nn.functional.dropout(x, p=0.5, training=False)`
  - `torch.nn.functional.normalize(x, dim=-1)`
  When in doubt, check the existing tests in the repo or the PyTorch docs for required arguments.
  Omitting mandatory arguments (e.g., `torch.nn.functional.softmax(x)` without `dim`) will cause
  runtime errors.
  Store `<torch_call>` (including any mandatory extra arguments) — it will be used in tests and
  benchmarks wherever the reference op is needed.
- `func_desc`: what the operator does
- `func_type`: one of the following categories (aligned with FlagGems test structure).
  **Auto-infer from the operator signature when possible:**
  - If 1 tensor input, no dim arg → `unary_pointwise`
  - If operator has >=2 tensor inputs, no dim arg → `binary_pointwise`
  - If has a `dim` argument → `reduction`
  - If name contains `norm`, `softmax` (also matches `log_softmax`), `rmsnorm`, or `batchnorm` → `normalization`
  - If name is `mm`, `matmul`, `bmm`, `addmm` → `blas`
  - Otherwise → `other`
  Confirm the inferred `func_type` with the user if ambiguous. Categories:
  - `unary_pointwise` — single-input elementwise ops (relu, sigmoid, abs, etc.)
  - `binary_pointwise` — two-input elementwise ops (add, mul, etc.)
  - `reduction` — ops that reduce dimensions (sum, mean, max, etc.)
  - `normalization` — norm ops (layer_norm, batch_norm, group_norm, etc.)
  - `blas` — matrix operations (matmul, mm, bmm, etc.)
  - `other` — everything else (indexing, special, etc.)
- `arg_names`, `arg_type`, `arg_descs`, `output_arg_desc`: parameter information

If the operator name is ambiguous, first check this common alias table before asking the user:

| User says | Correct `kernel_name` (torch.ops.aten) |
|---|---|
| swish | `silu` |
| negative / negate | `neg` |
| power | `pow` |
| square | `pow` (exp=2) |
| cube | `pow` (exp=3) |
| absolute | `abs` |
| multiply | `mul` |
| divide | `div` |
| clip | `clamp` |
| hard_swish | `hardswish` |
| hard_sigmoid | `hardsigmoid` |
| logarithm / ln | `log` |
| exponential | `exp` |
| hyperbolic_tangent | `tanh` |
| square_root | `sqrt` |
| cube_root | `cbrt` |
| inverse_sqrt | `rsqrt` |
| reciprocal / inverse | `reciprocal` |
| minimum | `min` |
| maximum | `max` |
| floor_divide | `floor_divide` |
| modulo / mod | `remainder` |

If the alias is not in this table, ask the user to clarify.

After determining `func_type`, derive the `<category>` value used for test/benchmark file paths:

| func_type | category (for file paths) |
|---|---|
| `unary_pointwise` | `unary_pointwise` |
| `binary_pointwise` | `binary_pointwise` |
| `reduction` | `reduction` |
| `normalization` | `norm` |
| `blas` | `blas` |
| `other` | `special` |

Store this `<category>` value — it determines which `tests/test_<category>_ops.py` and
`benchmark/test_<category>_perf.py` files to modify.

## Step 2: Check Whether the Operator Already Exists

Before calling the MCP generator, **thoroughly search** the codebase for existing implementations:

1. **Core ops**: Use the Glob tool to check for `src/flag_gems/ops/<kernel_name>.py`
2. **Experimental ops**: Use the Glob tool to check for `src/flag_gems/experimental_ops/<kernel_name>.py`
3. **Registration**: Use the Grep tool to search `src/flag_gems/ops/__init__.py` and `src/flag_gems/__init__.py` for the op name
4. **Aten registration**: Use the Grep tool to search for `torch.ops.aten.<kernel_name>` in the codebase (some registrations use the aten op reference instead of string names)
5. **Tests**: Use the Grep tool to search `tests/test_*_ops.py` for `test_accuracy_<kernel_name>`
6. **Benchmarks**: Use the Grep tool to search `benchmark/test_*_perf.py` for the op name in `forward_operations`

### If the operator already exists

Present findings to the user and ask them to choose one of the following:

**Option A — Skip generation**: The operator already exists; do nothing.

**Option B — Replace existing**: Overwrite the current implementation with MCP-generated code
(adapted to FlagGems conventions). This will modify:
  - `src/flag_gems/ops/<kernel_name>.py`
  - Potentially update tests and benchmarks

**Option C — Create a custom variant (side-by-side)**: Generate the operator under a different
name so it coexists with the original. The naming convention is `<kernel_name>_v2` (or a
user-specified suffix). This will:
  - Create `src/flag_gems/ops/<kernel_name>_v2.py`
  - Add a new import + `__all__` entry in `src/flag_gems/ops/__init__.py`
  - Do **NOT** register it in `_FULL_CONFIG` (it won't override the aten dispatch)
  - Create a standalone test in `experimental_tests/<kernel_name>_v2_test.py`
  - Include a perf benchmark in that same test file using `GenericBenchmark`

Only proceed to Step 3 after the user has made a choice.

### If the operator does NOT exist

Proceed directly to Step 3.

## Step 3: Research Context (flagos_wiki)

Before calling the MCP generator, use the Read tool to gather reference materials that improve generation quality:

1. **Read similar operator code** from the codebase. For example:
   - Unary pointwise → read `src/flag_gems/ops/abs.py` or `src/flag_gems/ops/sigmoid.py`
   - Binary pointwise → read `src/flag_gems/ops/add.py`
   - Reduction → read `src/flag_gems/ops/sum.py` or `src/flag_gems/ops/mean.py`
   - Normalization → read `src/flag_gems/ops/layer_norm.py`
   - BLAS → read `src/flag_gems/ops/mm.py`

2. **Read the test pattern** from the matching test file to understand shapes, dtypes, assertions.

3. **If replacing an existing op** (Option B), read the current implementation so the new version
   can be compared / improved upon.

4. Collect all findings and **summarize into concise notes** (not full file contents) to pass as
   the `flagos_wiki` parameter. For example:
   - `"abs.py uses pointwise_dynamic with DEFAULT promotion, returns tl.abs(x)"`
   - `"sigmoid.py uses INT_TO_FLOAT promotion, returns 1/(1+tl.exp(-x))"`
   - `"test pattern: @pytest.mark.xxx, POINTWISE_SHAPES, FLOAT_DTYPES, gems_assert_close"`

   If the `mcp__kernelgen-mcp__generate_operator` tool supports a `flagos_wiki` (or similarly named
   `context`/`references`) parameter, pass the collected notes. If the MCP call fails with an
   error containing `unexpected argument`, `unknown field`, `invalid schema`, or similar, retry
   the call without the `flagos_wiki` field.

## Step 4: Call kernelgen-mcp

Invoke `mcp__kernelgen-mcp__generate_operator` with the parameters gathered above, including the
`flagos_wiki` list for reference context.

**Set the iteration counter**: `iteration_count = 1`. This tracks total MCP calls for the final report.

The MCP returns four code blocks:
- `torch_code` — PyTorch reference implementation
- `triton_code` — Triton kernel implementation
- `test_func_code` — accuracy test code
- `benchmark_func_code` — performance benchmark code

## Step 5: Adapt and Place Code into the FlagGems Project

This is the most critical step. The generated code must be transformed to match FlagGems conventions.
The exact placement depends on the user's choice in Step 2.

**Transformation rules for pointwise ops**: The MCP generator will almost always output raw
pointer-based Triton code. For core pointwise ops, you MUST **always** rewrite it into the
`pointwise_dynamic` elementwise style:
- The `@triton.jit` kernel function receives **scalar elements** (not pointers)
- `pointwise_dynamic` handles all pointer arithmetic, tiling, and masking automatically
- The kernel only contains the elementwise math logic (e.g., `return tl.maximum(x, 0)`)
- **Remove ALL** `tl.load()`, `tl.store()`, pointer arithmetic (`ptr + offsets`), mask logic,
  and BLOCK_SIZE parameters — these are incompatible with `pointwise_dynamic`
- Only keep the pure math expression that transforms input element(s) to output element(s)

Examples of correct `pointwise_dynamic` kernels (keep ONLY the scalar `tl.*` math):
```python
# relu: return tl.maximum(x, 0)
# exp:  return tl.exp(x)
# silu: return x * tl.sigmoid(x)
# add:  return x + y
```

Do NOT keep `offsets`, `mask`, `n_elements`, `BLOCK_SIZE`, or any tiling logic.

**CRITICAL**: The MCP Triton code MUST NOT be copied directly for core ops. Always rewrite it
following FlagGems `pointwise_dynamic` conventions. Never trust the raw MCP code — it will
almost always be pointer-style which is incompatible with `pointwise_dynamic`.

---

### Path 1: New operator / Replace existing (Option B or new op)

#### 5a. Operator Implementation → `src/flag_gems/ops/<kernel_name>.py`

Use the Write tool (new file) or Edit tool (replacing existing) to create the operator file
following FlagGems conventions:

**Unary pointwise template:**
```python
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=<PROMOTION_METHODS>)
@triton.jit
def <kernel_name>_forward(x):
    return ...


def <kernel_name>(A):
    logger.debug("GEMS <KERNEL_NAME> FORWARD")
    return <kernel_name>_forward(A)
```

**Binary pointwise template:**
```python
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=<PROMOTION_METHODS>)
@triton.jit
def <kernel_name>_forward(x, y):
    return ...


def <kernel_name>(A, B):
    logger.debug("GEMS <KERNEL_NAME> FORWARD")
    return <kernel_name>_forward(A, B)
```

Choose the correct template based on `func_type`. For non-pointwise ops (reduction, BLAS,
normalization), do NOT use either template — instead read a similar existing op and follow
its pattern.

**For in-place variants**: Read an existing in-place operator (e.g., `relu.py`) to find the
exact call pattern used in this repo, then follow it. Common patterns include:

```python
def <kernel_name>_(A):
    logger.debug("GEMS <KERNEL_NAME>_ FORWARD")
    return <kernel_name>_forward(A, out0=A)
```

Do NOT assume the in-place parameter name — it may be `out0=A`, `out=A`, or another convention.
Always verify by reading an existing in-place operator first.

Key conventions:
- Use `pointwise_dynamic` decorator from `flag_gems.utils` for pointwise ops
- Use `triton.jit` decorator
- The kernel function takes raw tensor elements (not pointers) — `pointwise_dynamic` handles the boilerplate
- **Unary ops**: kernel has one parameter `(x)`, e.g. `def relu_forward(x)`
- **Binary ops**: kernel has two parameters `(x, y)`, e.g. `def add_forward(x, y)`
- Promotion methods — **always read a similar existing operator first** and copy its promotion
  method. If no similar operator exists, use these guidelines as fallback:
  - `INT_TO_FLOAT`: ops that always produce float output (exp, log, sigmoid, tanh, sqrt, etc.)
  - `COMPLEX_TO_FLOAT`: ops that reduce complex to real (abs)
  - `DEFAULT`: ops that preserve input dtype (relu, neg, add, mul, clamp, etc.)
  - **Template placeholder** `<PROMOTION_METHODS>` must expand to the full list:
    - Unary ops: `[(0, "DEFAULT")]` or `[(0, "INT_TO_FLOAT")]`
    - Binary ops: `[(0, "DEFAULT"), (1, "DEFAULT")]` or `[(0, "INT_TO_FLOAT"), (1, "INT_TO_FLOAT")]`
  - **When in doubt, prefer reading the repo** — promotion behavior is subtle and repo-specific.
    Promotion rules in FlagGems are repo-specific and must never be guessed if a similar operator exists.
- The wrapper function takes `A` as the input tensor parameter name (NOT `self`)
- Include `logger.debug("GEMS <NAME> FORWARD")` in the wrapper
- For in-place variants, read an existing in-place op first and copy its exact call pattern
- For non-pointwise ops (reduction, BLAS, normalization, etc.), follow the specific pattern of similar existing ops — do NOT force the `pointwise_dynamic` pattern on them

#### 5b. Register the Operator

Use the Edit tool to make these changes:

1. **`src/flag_gems/ops/__init__.py`**: Add import and `__all__` entry (in alphabetical order):
   ```python
   from flag_gems.ops.<kernel_name> import <kernel_name>, <kernel_name>_
   ```

2. **`src/flag_gems/__init__.py`**: Add to `_FULL_CONFIG` tuple (in alphabetical order).
   **IMPORTANT**: First read the existing `_FULL_CONFIG` entries to match the exact registration
   format used by this repo. Different versions may use different styles:
   - String style: `("relu", relu)`
   - Aten op style: `(torch.ops.aten.relu.default, relu)`
   Copy the pattern from existing entries. Do NOT guess the format.
   Insert the new entry in **alphabetical order** — do not change the order of existing entries.

#### 5c. Accuracy Test → `tests/test_<category>_ops.py`

Do NOT create a new test file. Use the Edit tool to add the test function to the appropriate existing test file:
- Unary pointwise: `tests/test_unary_pointwise_ops.py`
- Binary pointwise: `tests/test_binary_pointwise_ops.py`
- Reduction: `tests/test_reduction_ops.py`
- BLAS: `tests/test_blas_ops.py`
- Norm: `tests/test_norm_ops.py`
- Special: `tests/test_special_ops.py`

Follow the existing test pattern:
```python
@pytest.mark.<kernel_name>
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_<kernel_name>(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = <torch_call>(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.<kernel_name>(inp)

    gems_assert_close(res_out, ref_out, dtype)
```

Key conventions:
- Use `flag_gems.device` not `"cuda"`
- Use `to_reference()` to create reference tensors
- Use `flag_gems.use_gems()` context manager for the FlagGems call
- Use `gems_assert_close` or `gems_assert_equal` for comparison
- Mark with `@pytest.mark.<kernel_name>`
- If the marker `<kernel_name>` is not already registered, find where markers are defined
  (check in this order):
  1. `pytest.ini` — look for `[pytest] markers`
  2. `pyproject.toml` — look for `[tool.pytest.ini_options] markers`
  3. `setup.cfg` — look for `[tool:pytest] markers`
  Add the marker to whichever file already has a markers section.
  If none of them has a markers section, create one in `pytest.ini`.
- Use `POINTWISE_SHAPES`, `FLOAT_DTYPES` etc. from `accuracy_utils`

#### 5d. Performance Benchmark → `benchmark/test_<category>_perf.py`

Do NOT create a new benchmark file. Read the target benchmark file first, then use the Edit tool
to append the new tuple into the existing `forward_operations` list (find the list definition and
insert in alphabetical order). Follow the exact tuple format already used in the file.
- Unary pointwise: `benchmark/test_unary_pointwise_perf.py`
- Binary pointwise: `benchmark/test_binary_pointwise_perf.py`
- Reduction: `benchmark/test_reduction_perf.py`
- Norm: `benchmark/test_norm_perf.py`
- BLAS: `benchmark/test_blas_perf.py`
- Special: `benchmark/test_special_perf.py`

For unary pointwise ops, simply add a tuple to `forward_operations`:
```python
("<kernel_name>", <torch_call>, FLOAT_DTYPES),
```

For inplace ops, **only** add to `forward_inplace_operations` if PyTorch provides an inplace version.
Verify by running `hasattr(torch, "<kernel_name>_")` or checking if `torch.<kernel_name>_` exists
(e.g., `torch.relu_` exists, but `torch.gelu_` and `torch.softmax_` do not).
If the inplace op requires extra mandatory arguments (e.g., `clamp_` needs `min`/`max`), verify
the signature via `import inspect; inspect.signature(torch.<kernel_name>_)` before adding it.
**Important**: Always use `torch.<kernel_name>_` for the inplace reference — do NOT derive from
`<torch_call>` (e.g., `torch.nn.functional.relu_` does not exist):
```python
("<kernel_name>_", torch.<kernel_name>_, FLOAT_DTYPES),
```

---

### Path 2: Custom variant side-by-side (Option C)

#### 5a. Operator Implementation → `src/flag_gems/ops/<kernel_name>_v2.py`

Use the Write tool to create the file. Use the **raw Triton pointer-based style** (same as
`experimental_ops/`), since this operator does NOT go through `pointwise_dynamic` dispatch.
Keep the MCP-generated Triton code mostly as-is but ensure it:
- Is self-contained (no flag_gems.utils imports needed)
- Has proper `@triton.jit` kernel with pointer args, mask, BLOCK_SIZE
- **Has a Python wrapper function** that computes the grid and launches the kernel.
  **Ensure all kernel meta parameters (BLOCK_SIZE, etc.) are passed** — either via
  `@triton.autotune` (which sets them automatically) or as explicit keyword arguments:
  ```python
  # Option A: with @triton.autotune on the kernel (BLOCK_SIZE auto-set)
  def <kernel_name>_v2(x: torch.Tensor) -> torch.Tensor:
      assert x.is_cuda, "Triton kernel requires CUDA tensor"
      assert x.is_contiguous(), "Input tensor must be contiguous"
      output = torch.empty_like(x)
      n_elements = x.numel()
      grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
      _<kernel_name>_v2_kernel[grid](x, output, n_elements)
      return output

  # Option B: without autotune (adaptive BLOCK_SIZE heuristic)
  def <kernel_name>_v2(x: torch.Tensor) -> torch.Tensor:
      assert x.is_cuda, "Triton kernel requires CUDA tensor"
      assert x.is_contiguous(), "Input tensor must be contiguous"
      output = torch.empty_like(x)
      n_elements = x.numel()
      if n_elements < 4096:
          BLOCK_SIZE = 256
      elif n_elements < 65536:
          BLOCK_SIZE = 512
      else:
          BLOCK_SIZE = 1024
      grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
      _<kernel_name>_v2_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
      return output
  ```
  Adapt the grid computation to match the kernel pattern (elementwise vs row-wise).
- Exports the wrapper function named `<kernel_name>_v2`

Alternatively, if the user prefers the `pointwise_dynamic` style, use that and just rename
the function to `<kernel_name>_v2`.

#### 5b. Registration (limited)

Use the Edit tool to make these changes:

1. **`src/flag_gems/ops/__init__.py`**: Add import and `__all__` entry:
   ```python
   from flag_gems.ops.<kernel_name>_v2 import <kernel_name>_v2
   ```
2. Do **NOT** add to `_FULL_CONFIG` in `src/flag_gems/__init__.py` — the variant must NOT
   override the aten dispatch for the original operator.

#### 5c. Standalone Test + Benchmark → `experimental_tests/<kernel_name>_v2_test.py`

Use the Write tool to create a single self-contained test file following the `experimental_tests/` pattern:

```python
import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.ops.<kernel_name>_v2 import <kernel_name>_v2 as gems_op

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, **kwargs)

from benchmark.performance_utils import GenericBenchmark


def to_reference(inp, upcast=False):
    ref_inp = inp.to("cpu") if hasattr(inp, "to") else inp
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.<kernel_name>_v2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_<kernel_name>_v2_accuracy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = <torch_call>(ref_inp)
    act_out = gems_op(inp)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.<kernel_name>_v2
def test_<kernel_name>_v2_perf():
    def input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        yield inp,

    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name="<kernel_name>_v2",
        torch_op=<torch_call>,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    return bench.run()
```

---

## Step 5.5: Pre-test Validation

Before running the full test suite, perform two quick checks:

### 5.5a. Lint / Format Check

If the repo uses linting tools (ruff, black, isort, flake8), run them on the changed files to
catch style issues early. Use the shell tool:

```bash
python -m ruff check src/flag_gems/ops/<kernel_name>.py --fix
python -m ruff format src/flag_gems/ops/<kernel_name>.py
```

If `ruff` is not installed, check for other formatters in `pyproject.toml` or `setup.cfg` and
use those. If no linter is configured, skip this step.

### 5.5b. Triton Compile Smoke Test

Triton kernel compile errors only surface on first invocation (JIT compilation).
This smoke test ensures JIT compilation happens before pytest runs, catching compile
errors early with clear diagnostics rather than buried in test output:

**For Path 1:**

Adapt the smoke test call based on `func_type`:
- **Unary**: `torch.<kernel_name>(inp)`
- **Binary**: `torch.<kernel_name>(inp, inp)` (pass the same tensor twice)
- **Reduction**: `torch.<kernel_name>(inp)` or `torch.<kernel_name>(inp, dim=-1)` if `dim` is required
- **Normalization / other**: use `<torch_call>` with appropriate arguments

Use `inspect.signature` to infer missing arguments (e.g., weight, bias, eps for layer_norm) before constructing the smoke test. This ensures all mandatory parameters are included with sensible defaults, avoiding runtime errors from incomplete function calls.

```bash
python -c "
import torch, flag_gems
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for shape in [(1,), (4,), (128,), (4, 128), (0,)]:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device) if shape == (0,) else torch.randn(shape, dtype=dtype, device=flag_gems.device)
        with flag_gems.use_gems():
            out = <SMOKE_TEST_CALL>  # e.g. torch.relu(inp) or torch.add(inp, inp)
        print(f'Smoke test passed: {dtype}, shape={shape}')
"
```

**For Path 2:**
```bash
python -c "
import torch, flag_gems
from flag_gems.ops.<kernel_name>_v2 import <kernel_name>_v2
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for shape in [(1,), (4,), (128,), (4, 128), (0,)]:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device) if shape == (0,) else torch.randn(shape, dtype=dtype, device=flag_gems.device)
        out = <kernel_name>_v2(inp)  # For binary: <kernel_name>_v2(inp, inp)
        print(f'Smoke test passed: {dtype}, shape={shape}')
"
```

The `(0,)` shape tests the empty-tensor edge case (numel==0), which often causes kernel crashes.

If this fails, read the error and fix the kernel before proceeding to Step 6.

## Step 6: Run Accuracy Tests

Run the accuracy tests for the newly added operator.
Assume the current working directory is the repository root.

**For Path 1 (new / replace):**
```bash
python -m pytest tests/test_<category>_ops.py -m <kernel_name> -v
```

**For Path 2 (custom variant):**
```bash
python -m pytest experimental_tests/<kernel_name>_v2_test.py -v -k "accuracy"
```

Report the results to the user. If tests fail, follow the **error classification and retry
protocol** below. This protocol strictly limits self-fix attempts to prevent infinite loops.

### Error Classification

First, classify the error into one of two categories:

**Category A — Compilation / Import errors** (model may attempt 1 self-fix):
- `ImportError`, `ModuleNotFoundError` — wrong import path or missing module
- `SyntaxError` — Python syntax issue
- `TritonCompilationError`, `CompilationError` — Triton kernel compile failure
- `NameError` — undefined variable or function
- `TypeError` in function call signature — wrong number/type of arguments
- `AttributeError` — wrong attribute access on a module or object

**Category B — Algorithm / Numerical accuracy errors** (do NOT self-fix):
- `AssertionError` from `torch.testing.assert_close` or `gems_assert_close` — numerical mismatch
- Wrong output values (results don't match reference)
- Shape mismatch in output tensors
- NaN or Inf in output
- Any error that indicates the kernel logic is incorrect

### Retry Protocol (strictly follow this order)

**Step 6a. (Category A only) Self-fix — maximum 1 attempt:**
If the error is Category A (compilation/import), you may attempt exactly ONE fix:
1. Use the Read tool to examine the error traceback carefully.
2. Apply a targeted fix using the Edit tool (e.g., fix import path, fix syntax, fix argument name).
3. Use the shell tool to re-run the tests.
4. If the test **passes** → proceed to Step 7.
5. If the test **still fails** → proceed to Step 6b. Do NOT attempt a second self-fix.

**If the error is Category B (algorithm/accuracy), skip Step 6a entirely** — go directly
to Step 6b. Do NOT attempt to fix algorithm logic yourself, as this typically leads to
an endless fix-retry loop without converging. The MCP service has better optimization
capabilities for these issues.

**Step 6b. MCP re-generation — pass error context to generate_operator:**
Re-call `mcp__kernelgen-mcp__generate_operator` with the **same parameters as Step 4**,
but add the error information to `flagos_wiki` as additional hints.
**Increment**: `iteration_count += 1`.
Keep `flagos_wiki` concise — maximum 10 items total. If retrying multiple times, replace
earlier error entries rather than appending, to avoid bloating the prompt.
```python
flagos_wiki = [
    # ... original flagos_wiki items from Step 3 ...
    "Previous generation failed accuracy test: <brief error description>",
    "Error was: <key line from traceback>",
    "Fix hint: <your analysis, e.g. 'promotion method should be INT_TO_FLOAT not DEFAULT'>"
]
```
Replace the kernel code with the new MCP output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6c.

**Step 6c. MCP optimization — pass error context to optimize_triton:**
Try `mcp__kernelgen-mcp__optimize_triton` with the current kernel code and the
`check_result` parameter containing the error traceback.
**Increment**: `iteration_count += 1`. This endpoint can fix
memory access patterns, index calculations, and numerical issues.
Replace the kernel code with the optimized output, re-run tests.
- If tests **pass** → proceed to Step 7.
- If tests **still fail** → proceed to Step 6d.

**Step 6d. Stop and report:**
Do not keep retrying. Report the failure to the user with:
- The specific test failures and error messages
- Your analysis of what might be wrong
- Suggestion to try with different `func_type` or additional `flagos_wiki` hints

## Step 7: Run Performance Benchmark

Run the performance benchmark.

**For Path 1 (new / replace):**
```bash
python -m pytest benchmark/test_<category>_perf.py -m <kernel_name> -v
```

**For Path 2 (custom variant):**
```bash
python -m pytest experimental_tests/<kernel_name>_v2_test.py -v -k "perf"
```

Look for lines in the output containing keywords like `speedup`, `latency`, `gems`, `torch`, or
timing values. Use regex patterns to extract numbers, accounting for various formats:
- `(\d+(?:\.\d+)?)\s*(us|ms|s)` — number with space before unit
- `(\d+(?:\.\d+)?)(ms|us|s)` — number directly attached to unit (e.g., `0.123ms`)
- `(\d+(?:\.\d+)?)\s*(us|ms|s)/iter` — number with per-iteration suffix (e.g., `0.123 ms/iter`)
- `time:\s*(\d+(?:\.\d+)?)` — labeled timing values
- `(?i)avg:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — average timing value (e.g., `avg: 0.123 ms`)
- `(?i)mean:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — mean timing value (pytest-benchmark format, e.g., `mean: 0.123 ms`)
- `(?i)median:\s*(\d+(?:\.\d+)?)\s*(us|ms|s)` — median timing value
- `(?i)Torch.*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — PyTorch reference time (case-insensitive)
- `(?i)GEMS.*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — FlagGems kernel time (case-insensitive)
- `(?i)(flag_gems|gems).*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — FlagGems kernel time (alternate)
- `(?i)(torch|pytorch).*?(\d+(?:\.\d+)?)\s*(us|ms|s)` — PyTorch time (case-insensitive)

**Do not just copy the raw table** — always compute and report actual speedup ratios.

If a speedup metric is printed directly, extract and report it. Otherwise compute:
`speedup = torch_latency / gems_latency` (>1.0 means gems is faster).

**Note**: Triton kernels are JIT-compiled on first invocation, which adds significant overhead.
If benchmark results look unreasonably slow on the first config, check whether the benchmark
framework includes a warmup phase. If not, the first data point may be an outlier — **if the
first timing sample is >5x larger than the median of remaining samples, exclude it** from the
average speedup calculation and note "first sample excluded (Triton JIT compile overhead)" in
the report. Triton JIT compilation is detected when the first invocation is disproportionately
slow — always ignore such outliers when computing average speedup.

## Step 8: Summary

Provide a clear summary to the user with **exact numbers** extracted from test and benchmark
output:

```
=== KernelGen Operator Generation Report ===

Operator Name: <kernel_name>
Generation Mode: New / Replace Existing / Custom Variant (v2)
Operator Type: <func_type>

File Changes:
  - [New/Modified] src/flag_gems/ops/<kernel_name>.py
  - [Modified] src/flag_gems/ops/__init__.py
  - [Modified] src/flag_gems/__init__.py
  - [Modified] tests/test_<category>_ops.py
  - [Modified] benchmark/test_<category>_perf.py

Accuracy Tests: <N> passed, <M> failed (total <N+M> test cases)
  Pass Rate: <N/(N+M)*100>%
  Failed Cases: <list failed test names if any, or "None">

Performance Benchmark:
  Avg Speedup: <X.XX>x vs PyTorch reference
  Best Speedup: <X.XX>x (shape=<S>, dtype=<D>)
  Worst Speedup: <X.XX>x (shape=<S>, dtype=<D>)
  (If benchmark was not run or failed, write "Incomplete" and explain why)

MCP Iterations: <iteration_count> (write 1 if first generation passed)

Performance Analysis:
  <Assess based on average speedup:
   - Speedup > 5.0x → "Extreme fusion success, typical for fused multi-op kernels"
   - Speedup > 3.0x → "Excellent compute-bound optimization, significant kernel fusion benefit"
   - Speedup > 2.0x → "Compute-bound optimization effective, significant speedup achieved"
   - Speedup 1.2x ~ 2.0x → "Moderate speedup, kernel likely balanced between compute and memory"
   - Speedup 0.5x ~ 1.2x → "Kernel likely memory-bound, limited optimization headroom"
   - Speedup < 0.5x → "Kernel slower than PyTorch reference — likely suboptimal memory access pattern">

Issues and Fixes: (if any)
```

**How to extract the numbers:**
- **Pass/fail counts**: Parse pytest output for the line matching `X passed` or `X passed, Y failed`.
  Use regex pattern `(\d+) passed` and `(\d+) failed` to extract.
- **Speedup**: Parse benchmark output for FlagGems vs PyTorch time values. Calculate
  `speedup = torch_latency / gems_latency` for each config. Report min, max, and average.
  - **Do not just copy the raw table** — always compute and report the actual speedup ratios.

**After presenting the summary, check the speedup:**

- **If average speedup < 0.5x** (kernel slower than PyTorch), proactively warn the user:
  ```
  ⚠️ The generated Triton kernel is slower than the PyTorch reference implementation.
  This is typically caused by suboptimal memory access patterns or improper block size configuration.
  Would you like to use /kernelgen_optimizer to optimize this kernel's performance?
  ```

- **If average speedup is 0.5x ~ 1.2x**, ask the user:
  ```
  Current speedup is low (<X.XX>x), there may be room for optimization.
  Would you like to use /kernelgen_optimizer to try improving performance?
  ```

- **If average speedup > 1.2x**, no action needed — report the result normally.

## Important Notes

- **`kernel_name` must match the `torch` API name** (e.g., `silu` not `swish`, `neg` not `negate`).
  The accuracy test calls `torch.<kernel_name>(...)`, so a mismatch will cause test failures.
- **Never overwrite existing operators** without explicit user permission (Step 2 choice)
- **Always use `pointwise_dynamic`** for core pointwise ops — do NOT write raw pointer-based Triton kernels
  for core ops. Raw pointer style is only for custom variants (Option C / Path 2).
- **Follow alphabetical ordering** when adding entries to `__init__.py` and `__all__`
- **Match the existing code style** exactly — read existing files first and copy the pattern
  (imports, logging, naming conventions, registration format)
- **Benchmark follow the existing pattern** in the file — do NOT invent new patterns for
  `forward_operations` or `forward_inplace_operations`
- If accuracy tests fail, try to fix the operator. If benchmark is slow, consider calling
  `mcp__kernelgen-mcp__optimize_triton` to optimize the kernel
- When creating a custom variant, always make it clear that it does NOT override the aten dispatch
  to avoid conflicts with the existing operator
- **Speedup formula**: `speedup = torch_latency / gems_latency` (values > 1.0 mean gems is faster)
- **Always use built-in tools** (shell, Read, Write, Edit, Grep, Glob) instead of
  outputting commands or code snippets for the user to execute manually
