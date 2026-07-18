#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PR Gate Check — 在 git push / gh pr create 前强制验证
# 被 .claude/settings.json 的 PreToolUse hook 调用
#
# 检查逻辑：
# 1. 当前分支是否是 pr/<op> 格式
# 2. check_operator.py 是否通过（检查 .pr_gate_passed 标记文件）
# 3. 如果没通过，输出 JSON 阻止操作

REPO_DIR="${FLAGGEMS_REPO:-/workspace/FlagGems_minimax_2_7_pr}"
SCRIPTS_DIR="${FLAGGEMS_SCRIPTS_DIR:-/workspace/.claude/skills/flaggems-pr-submit/scripts}"
GATE_DIR="$REPO_DIR/.pr_gate"

# 获取当前分支名
BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# 不是 pr/ 分支就放行
if [[ ! "$BRANCH" =~ ^pr/ ]]; then
    exit 0
fi

OP_NAME="${BRANCH#pr/}"
GATE_FILE="$GATE_DIR/$OP_NAME.passed"

# 检查 gate 标记
if [[ -f "$GATE_FILE" ]]; then
    FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$GATE_FILE" 2>/dev/null || echo 0) ))
    if [[ $FILE_AGE -lt 3600 ]]; then
        # 确认测试已通过（不允许 test=skip 绕过）
        if grep -q "test=pass" "$GATE_FILE"; then
            exit 0  # 放行
        fi
    fi
fi

# 阻止：输出 JSON
cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "PR Gate: check_operator.py 未通过。请先运行: python $SCRIPTS_DIR/submit_operator.py $OP_NAME --repo-dir $REPO_DIR"
  }
}
EOF
exit 0
