---
name: skill-test
description: A test skill for CI validation testing purposes
---

# Skill Test

This is a test skill created to verify that the CI validation pipeline correctly catches naming violations.

## Usage

Use this skill to test the validation script.

## Example

```
# This skill is intentionally missing the -flagos suffix
# Expected: validation should fail
```

## Troubleshooting

If validation passes, the suffix check is not working correctly.
