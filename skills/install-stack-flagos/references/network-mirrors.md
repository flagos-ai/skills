# Network & Mirror Configuration

When running in China mainland, direct access to GitHub and PyPI may be slow or
blocked. Use `scripts/detect_network.py` to auto-detect, or apply these rules.

## Detection

Run `scripts/detect_network.py` inside the container. It probes both endpoints
and returns JSON with mirror config.

## GitHub Mirror

| Scenario | GITHUB_PREFIX |
|----------|--------------|
| Direct OK | `https://github.com` |
| China mainland | `https://ghfast.top/https://github.com` |

Usage: `git clone ${GITHUB_PREFIX}/<org>/<repo>`

## PyPI Mirror

| Scenario | PIP_INDEX |
|----------|-----------|
| Direct OK | *(empty)* |
| China mainland | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

Usage: `pip install ${PIP_INDEX} <package>`

Persistent config (inside container):
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## FlagOS PyPI (for FlagTree)

Always use FlagOS PyPI for FlagTree regardless of network:
```
--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
--trusted-host=resource.flagos.net
```

## Rules

Every command that accesses GitHub or PyPI MUST:
1. Use the configured mirror/proxy prefix
2. Have a timeout (no hanging on bad network)
3. On failure, report whether it's a network issue or a real error
