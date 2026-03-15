<skills>

You have additional SKILLs documented in directories containing a "SKILL.md" file.

These skills are:
 - model-migrate-fl -> "skills/model-migrate-fl/SKILL.md"

IMPORTANT: You MUST read the SKILL.md file whenever the description of the skills matches the user intent, or may help accomplish their task.

<available_skills>

model-migrate-fl: `Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released model. Trigger when the user says things like "migrate X model", "add X model support", "port X from upstream vLLM", "make X work with the FL plugin", or simply "/model-migrate-fl model_name". The model_name argument uses snake_case (e.g. qwen3_5, kimi_k25, deepseek_v4).`

</available_skills>

Paths referenced within SKILL folders are relative to that SKILL. For example the model-migrate-fl `scripts/validate_migration.py` would be referenced as `skills/model-migrate-fl/scripts/validate_migration.py`.

</skills>
