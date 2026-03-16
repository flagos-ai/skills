# JSON Schemas

Defines the JSON schemas used by skill-creator for evaluation and testing.

---

## evals.json

Defines test cases for a skill. Located at `evals/evals.json` within the skill directory.

```json
{
  "skill_name": "example-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "User's example prompt",
      "expected_output": "Description of expected result",
      "files": ["evals/files/sample1.txt"],
      "assertions": [
        "The output includes X",
        "Step Y was executed correctly",
        "No errors were reported"
      ]
    }
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `skill_name` | Yes | Must match the skill's frontmatter `name` |
| `evals[].id` | Yes | Unique integer identifier |
| `evals[].prompt` | Yes | The user prompt to test with |
| `evals[].expected_output` | Yes | Human-readable description of success |
| `evals[].files` | No | Input file paths relative to skill root |
| `evals[].assertions` | No | List of verifiable statements to check |

### Writing Good Test Prompts

- Use realistic language a real user would actually say
- Include both direct invocations (`/skill-name arg`) and natural language triggers
- Cover happy paths, edge cases, and error scenarios
- At least 2-3 test prompts per skill

### Writing Good Assertions

- Make assertions objectively verifiable (not subjective)
- Use descriptive text that reads clearly in reports
- Check both positive outcomes ("file was created") and negative ("no error messages")
- Don't assert on implementation details — focus on user-visible outcomes

---

## eval_result.json

Output from running evaluations. Located at `evals/results/eval_result.json`.

```json
{
  "skill_name": "example-skill",
  "timestamp": "2026-03-15T10:30:00Z",
  "results": [
    {
      "id": 1,
      "prompt": "User's example prompt",
      "assertions": [
        {
          "text": "The output includes X",
          "passed": true,
          "evidence": "Found X in the generated output at line 15"
        },
        {
          "text": "Step Y was executed correctly",
          "passed": false,
          "evidence": "Step Y was skipped due to missing dependency"
        }
      ],
      "summary": {
        "passed": 1,
        "failed": 1,
        "total": 2,
        "pass_rate": 0.5
      }
    }
  ],
  "overall": {
    "total_evals": 1,
    "total_assertions": 2,
    "total_passed": 1,
    "overall_pass_rate": 0.5
  }
}
```

---

## validation_result.json

Output from the validation script. Located at `evals/results/validation_result.json`.

```json
{
  "skill_name": "example-skill",
  "skill_path": "/path/to/skills/example-skill",
  "timestamp": "2026-03-15T10:30:00Z",
  "valid": true,
  "errors": [],
  "warnings": [
    "No README.md found (recommended)",
    "Script scripts/helper.py lacks execute permission"
  ],
  "checks": {
    "frontmatter": "pass",
    "name_convention": "pass",
    "description_quality": "pass",
    "body_content": "pass",
    "file_references": "pass",
    "script_permissions": "warning"
  }
}
```
