# Workflow and Templates

This file provides executable templates for TLE delivery.
Use with `references/tle-sources.md`.

## 1. Standard Workflow

### 1) Intake
1. Capture goal, non-goal, acceptance.
2. Confirm scope by layer (API/verifier/lowering/pipeline/tests).
3. Confirm owner and milestone.

### 2) Design
1. Write the behavior contract in plain language.
2. List exact files likely to change.
3. Define failure modes and rollback plan.

### 3) Implementation
1. Land smallest safe patch first.
2. Keep semantic checks near API/verifier entry.
3. Keep backend coupling explicit.

### 4) Validation
1. Run targeted unit tests first.
2. Run targeted integration tests.
3. Run backend-specific tests if needed.
4. Run build/check tasks for compiler-level changes.

### 5) Handoff
1. Fill Fix Summary.
2. Fill Lessons Entry.
3. Fill Merge Decision Package.

## 2. Environment Preflight Template

```text
Environment
- Python env: <system python / venv / conda env>
- Backend: <nvidia / other>
- Build dir: <path>
- Build entrypoint: <repo-specific command; do not assume a fixed script name>

Preflight Commands
- <command to verify python/triton/torch>
- <command to verify cuda device>
- <command to build if needed>

Status
- <pass/fail + notes>
```

## 3. Requirement Intake Template

```text
# Requirement Intake

Goal:
- <what must be achieved>

Non-goal:
- <out of scope>

Acceptance:
- <testable condition 1>
- <testable condition 2>

Impact Scope:
- API:
- Verifier:
- Lowering:
- Pipeline:
- Tests:

Risks:
- <risk 1>
- <risk 2>

Execution Plan:
1. <step>
2. <step>
3. <step>
```

## 4. Validation Matrix Template

```text
Targeted Unit
- <command>
  - <result>

Targeted Integration
- <command>
  - <result>

Backend-Specific
- <command>
  - <result>

Build/Check
- <command>
  - <result>
```

## 5. Performance Record Template

```text
Benchmark Goal
- <metric and target>

Fixed Setup
- Shape:
- Seed:
- Launch config:
- Command:

Baseline
- Correctness:
- Metric value:

Change Under Test
- One-line hypothesis:
- One-line code change:

After
- Correctness:
- Metric value:
- Delta:

Evidence
- TTGIR/PTX marker changes:

Decision
- <keep/revert + reason>
```

## 6. Fix Summary Template

```text
Root Cause
- <one sentence on triggering mechanism>

Changes
- <path:line> - <key change>
- <path:line> - <key change>

Validation
- <command>
  - <result>
- <command>
  - <result>

Risk and Follow-up
- Risk: <None or residual risk>
- Follow-up: <None or next tasks>
```

## 7. Lessons Entry Template

```text
Context
- Scope: <component/kernel/pass>
- Trigger: <bug/perf regression/new optimization>
- Date: <YYYY-MM-DD>

Process (Condensed)
1. Repro with smallest stable case.
2. Capture IR/PTX evidence.
3. Apply minimal safe change.
4. Re-run targeted tests, then broader suite.
5. Re-benchmark with fixed command.

Root Cause
- <one sentence>

What Changed
- <path:line> - <key change>
- <path:line> - <key change>

Validation
- <command>
  - <result>
- <command>
  - <result>

Keep / Avoid
- Keep: <rule>
- Avoid: <anti-pattern>

Follow-up
- <next optimization or cleanup>
```

## 8. Merge Decision Package Template

```text
Change Summary by Layer
- API:
- Verifier:
- Lowering:
- Pipeline:
- Tests:

Validation Evidence
- <command/result list>

Performance Impact
- <none / summary>

Residual Risks
- <None or list>

Recommended Follow-ups
1. <item>
2. <item>
```
