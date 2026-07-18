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

The canonical Agent Skills specification is maintained at:

**https://agentskills.io/specification**

This file serves as a redirect. Do not duplicate the spec here — always refer to the canonical source.

## Quick Reference

### Required Frontmatter

```yaml
---
name: my-skill-name          # ≤64 chars, lowercase+hyphens, must match directory name
description: >                # ≤1024 chars, what + when
  One or two sentences explaining what this skill does
  and when it should be triggered.
---
```

### Optional Frontmatter

```yaml
license: "Apache-2.0"        # or "Proprietary. See LICENSE.txt"
compatibility: "Python 3.10+" # ≤500 chars, environment requirements
metadata:
  version: "1.0.0"
  author: "flagos-ai"
  category: "workflow-automation"
  tags: [tag1, tag2]
```

### Directory Layout

```
my-skill-name/
├── SKILL.md          # Required
├── LICENSE.txt        # Recommended
├── references/        # Optional: detailed docs
├── scripts/           # Optional: executable scripts
├── assets/            # Optional: icons, templates
└── examples/          # Optional: usage examples
```
