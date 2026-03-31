# KernelGen MCP Configuration Check & Auto-Setup

This file is responsible for checking whether the `kernelgen-mcp` MCP service is configured,
and guiding the user through automatic configuration if it is not.
Before any sub-skill (generate / optimize / specialize) executes, `SKILL.md` dispatches to
this file to ensure MCP is ready before proceeding to subsequent workflows.

---

## Step 1: Check Whether MCP Is Already Configured

Use the Read tool to check the following files in order (only check project-local paths — do not read the user's home directory):

1. `.mcp.json`
2. `.claude/settings.json`

For each file:
- If the file does not exist, skip it
- If the file exists, parse the JSON and check whether `mcpServers` contains a key that includes `kernelgen` (case-insensitive)

**Decision rules**:
- Found in any file → **MCP is configured**, return immediately and continue the workflow
- Not found in either file → **MCP is not configured**, proceed to Step 2

---

## Step 2: Guide the User to Obtain a Token

Output the following message to the user:

```
The KernelGen MCP toolset is not yet configured.

Please follow these steps:
1. Visit https://kernelgen.flagos.io/mcp to register and obtain your KernelGen Token
2. Paste the KernelGen Token here, and I will complete the configuration automatically

(You only need to provide the KernelGen Token — the MCP service URL is built-in and does not need to be entered separately)
```

Wait for the user to provide the Token.

---

## Step 3: Auto-Write Configuration

After the user provides the Token, write the configuration using the following logic:

**Target configuration format** (written to `.mcp.json`):

```json
{
  "mcpServers": {
    "kernelgen-server": {
      "type": "sse",
      "url": "https://kernelgen.flagos.io/sse/",
      "headers": {
        "Authorization": "Bearer <USER_TOKEN>"
      }
    }
  }
}
```

**Write logic**:

1. Use the Read tool to check whether `.mcp.json` already exists
2. **If the file already exists**:
   - Read and parse the JSON
   - If the `mcpServers` key already exists, merge `"kernelgen-server": {...}` into it — **do not delete any other existing MCP service entries**
   - If the `mcpServers` key does not exist, create it and add the `kernelgen-server` entry
   - Use the Edit tool to update the file
3. **If the file does not exist**:
   - Use the Write tool to create the file with the complete JSON above

**Important notes**:
- The MCP service URL is fixed as `https://kernelgen.flagos.io/sse/` — the user does not need to provide it
- The key name uses `kernelgen-server` (consistent with the MCP tool names `mcp__kernelgen-server__*`)
- Never overwrite other configuration entries in the file

---

## Step 4: Prompt the User to Restart

After the configuration is written, output the following to the user:

```
MCP configuration has been written to .mcp.json. Please restart Claude Code for the configuration to take effect, then re-run the command.
```

**Stop here** — do not continue executing subsequent sub-skill workflows. When the user restarts and re-triggers the skill, Step 1 will detect that the configuration already exists and pass through directly.
