#!/usr/bin/env python3

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

"""检查多重载算子的 yaml id / benchmark mark+op_name / test mark 三方一致性。

查找逻辑：传入算子名，通过 yaml 的 id 前缀匹配和 for 字段关联两个条件的并集
来确定所有相关重载条目，然后逐一检查 benchmark/test 中的对齐情况。

用法:
    python check_overload_consistency.py <operator_name> [--repo-dir /path/to/FlagGems]

示例:
    python check_overload_consistency.py reflection_pad3d
    python check_overload_consistency.py eq --repo-dir /workspace/FlagGems
    python check_overload_consistency.py max
"""

import argparse
import os
import re
import sys

import yaml


class Colors:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg):
    print(f"  {Colors.OK}✓{Colors.END} {msg}")


def warn(msg):
    print(f"  {Colors.WARN}⚠{Colors.END} {msg}")


def fail(msg):
    print(f"  {Colors.FAIL}✗{Colors.END} {msg}")


def section(msg):
    print(f"\n{Colors.BOLD}[{msg}]{Colors.END}")


BUILTIN_MARKS = frozenset(
    ("parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings")
)


def get_op_id(op_name):
    return op_name.lstrip("_")


def find_related_yaml_ids(yaml_path, op_name, op_id):
    """通过 id 前缀匹配 + for 字段关联的并集找到所有相关 yaml 条目。"""
    if not os.path.isfile(yaml_path):
        return {}

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        fail(f"YAML 解析失败: {e}")
        return {}

    if not data:
        return {}

    related = {}
    for op in data.get("ops", []):
        entry_id = op.get("id", "")
        for_list = op.get("for", [])

        # 条件 A: id 精确匹配
        match_a = (entry_id == op_id)

        # 条件 B: for 列表中有以 op_name 开头的 aten name
        match_b = any(
            item == op_name or item.startswith(op_name + ".")
            for item in for_list
        )

        if match_a or match_b:
            related[entry_id] = op

    return related


def extract_func_info(file_path):
    """提取文件中每个 test 函数的 marks 和 op_name。"""
    if not os.path.isfile(file_path):
        return []

    try:
        with open(file_path) as f:
            content = f.read()
    except OSError:
        return []

    lines = content.split("\n")
    results = []

    # 预扫描：收集所有 @pytest.mark.<name> 的行号和名称
    mark_lines = {}
    for i, line in enumerate(lines):
        m = re.match(r"\s*@pytest\.mark\.(\w+)", line)
        if m and m.group(1) not in BUILTIN_MARKS:
            mark_lines[i] = m.group(1)

    # 找到所有 def test_ 行的位置
    func_positions = []
    for i, line in enumerate(lines):
        if re.match(r"^def\s+(test_\w+)\s*\(", line):
            func_positions.append(i)

    # 对每个函数，将"上一个函数定义行"到"当前函数定义行"之间的 marks 归属于它
    for idx, func_line in enumerate(func_positions):
        match = re.match(r"^def\s+(test_\w+)\s*\(", lines[func_line])
        func_name = match.group(1)

        # 区间上界：上一个 def 行（或文件开头）
        prev_boundary = func_positions[idx - 1] + 1 if idx > 0 else 0

        # 收集区间内的 marks
        marks = set()
        for mark_line, mark_name in mark_lines.items():
            if prev_boundary <= mark_line < func_line:
                marks.add(mark_name)

        # 向下找 op_name
        op_name = None
        for k in range(func_line + 1, min(func_line + 30, len(lines))):
            if re.match(r"^(def |class )", lines[k]):
                break
            op_match = re.search(r'op_name\s*=\s*["\'](\w+)["\']', lines[k])
            if op_match:
                op_name = op_match.group(1)
                break

        results.append({
            "func_name": func_name,
            "marks": marks,
            "op_name": op_name,
            "line": func_line + 1,
        })

    return results


def check_consistency(yaml_ids, bench_info, test_info, op_id):
    """交叉验证 yaml ids / benchmark marks+op_name / test marks。"""
    errors = []
    warnings = []

    section("YAML → Benchmark/Test 对齐检查")

    for yaml_id in sorted(yaml_ids):
        print(f"\n  yaml id: {Colors.BOLD}{yaml_id}{Colors.END}")

        # Benchmark mark
        bench_funcs_with_mark = [
            info for info in bench_info if yaml_id in info["marks"]
        ]
        if bench_funcs_with_mark:
            func_names = [f["func_name"] for f in bench_funcs_with_mark]
            ok(f"benchmark @pytest.mark.{yaml_id} → {', '.join(func_names)}")
        else:
            errors.append(
                f"yaml id '{yaml_id}' 在 benchmark 中无对应 @pytest.mark.{yaml_id}"
            )
            fail(f"benchmark 缺少 @pytest.mark.{yaml_id}")

        # Benchmark op_name
        for info in bench_funcs_with_mark:
            if info["op_name"] is None:
                warnings.append(
                    f"benchmark {info['func_name']} 未找到 op_name"
                )
                warn(f"{info['func_name']}: 未找到 op_name")
            elif info["op_name"] == yaml_id:
                ok(f"{info['func_name']}: op_name=\"{info['op_name']}\" ✓")
            else:
                errors.append(
                    f"benchmark {info['func_name']}: "
                    f"op_name=\"{info['op_name']}\" ≠ yaml id '{yaml_id}'"
                )
                fail(
                    f"{info['func_name']}: op_name=\"{info['op_name']}\" "
                    f"≠ yaml id \"{yaml_id}\""
                )

        # Test mark
        test_funcs_with_mark = [
            info for info in test_info if yaml_id in info["marks"]
        ]
        if test_funcs_with_mark:
            func_names = [f["func_name"] for f in test_funcs_with_mark]
            ok(f"test @pytest.mark.{yaml_id} → {', '.join(func_names)}")
        else:
            if yaml_id != op_id:
                warnings.append(
                    f"yaml id '{yaml_id}' 在 test 中无对应 mark（建议对齐）"
                )
                warn(f"test 缺少 @pytest.mark.{yaml_id}（建议对齐）")
            else:
                errors.append(
                    f"yaml id '{yaml_id}' 在 test 中无对应 @pytest.mark.{yaml_id}"
                )
                fail(f"test 缺少 @pytest.mark.{yaml_id}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="检查多重载算子的 yaml/benchmark/test 一致性"
    )
    parser.add_argument("operator", help="算子名称（如 reflection_pad3d）")
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get("FLAGGEMS_REPO", "/workspace/FlagGems_minimax_2_7_pr"),
        help="FlagGems 仓库路径",
    )
    args = parser.parse_args()

    op_name = args.operator
    op_id = get_op_id(op_name)
    repo = os.path.abspath(args.repo_dir)

    yaml_path = os.path.join(repo, "conf/operators.yaml")
    bench_path = os.path.join(repo, f"benchmark/test_{op_id}.py")
    test_path = os.path.join(repo, f"tests/test_{op_id}.py")

    print(f"{Colors.BOLD}多重载一致性检查: {op_name}{Colors.END}")
    print(f"仓库: {repo}")

    # 1. 查找相关 yaml ids
    section("YAML 条目扫描")
    yaml_entries = find_related_yaml_ids(yaml_path, op_name, op_id)

    if not yaml_entries:
        fail(f"yaml 中未找到与 '{op_id}' 相关的条目")
        sys.exit(1)

    for entry_id, entry in yaml_entries.items():
        ok(f"id: {entry_id} (for: {entry.get('for', [])})")

    if len(yaml_entries) == 1:
        ok("仅一个 yaml id，检查其对齐情况")
    else:
        print(f"\n  发现 {len(yaml_entries)} 个相关条目")

    # 2. 扫描 benchmark
    section("Benchmark 文件扫描")
    if not os.path.isfile(bench_path):
        fail(f"文件不存在: {bench_path}")
        sys.exit(1)

    bench_info = extract_func_info(bench_path)
    for info in bench_info:
        marks_str = ", ".join(sorted(info["marks"])) or "(无 mark)"
        op_str = info["op_name"] or "(无)"
        ok(f"{info['func_name']}: mark=[{marks_str}], op_name={op_str}")

    # 3. 扫描 test
    section("Test 文件扫描")
    if not os.path.isfile(test_path):
        warn(f"文件不存在: {test_path}（跳过 test mark 检查）")
        test_info = []
    else:
        test_info = extract_func_info(test_path)
        for info in test_info:
            marks_str = ", ".join(sorted(info["marks"])) or "(无 mark)"
            ok(f"{info['func_name']}: mark=[{marks_str}]")

    # 4. 交叉验证
    errors, warnings = check_consistency(
        yaml_entries.keys(), bench_info, test_info, op_id
    )

    # 5. 汇总
    section("检查汇总")
    print(f"\n  算子: {op_name}")
    print(f"  yaml 条目数: {len(yaml_entries)}")
    print(f"  错误: {Colors.FAIL}{len(errors)}{Colors.END}")
    print(f"  警告: {Colors.WARN}{len(warnings)}{Colors.END}")

    if errors:
        print(f"\n  {Colors.FAIL}{Colors.BOLD}必须修复:{Colors.END}")
        for e in errors:
            print(f"    - {e}")

    if warnings:
        print(f"\n  {Colors.WARN}{Colors.BOLD}建议修复:{Colors.END}")
        for w in warnings:
            print(f"    - {w}")

    if not errors:
        print(f"\n  {Colors.OK}{Colors.BOLD}所有一致性检查通过 ✓{Colors.END}")
    else:
        print(f"\n  {Colors.FAIL}{Colors.BOLD}存在 {len(errors)} 个一致性问题 ✗{Colors.END}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
