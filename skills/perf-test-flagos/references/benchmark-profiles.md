# Benchmark Profiles

Default 5-profile matrix covering different workload patterns.

## Default Profiles

| Profile | Input Len | Output Len | Num Prompts | Purpose |
|---------|-----------|------------|-------------|---------|
| short_prefill_short_decode | 256 | 128 | 100 | Latency-sensitive chat |
| long_prefill_short_decode | 4096 | 128 | 100 | Summarization / RAG |
| short_prefill_long_decode | 256 | 1024 | 100 | Long-form generation |
| long_prefill_long_decode | 4096 | 1024 | 100 | Heavy workload |
| high_concurrency | 1024 | 512 | 500 | Throughput under load |

## Key Metrics

| Metric | Description | Lower is better? |
|--------|-------------|:-:|
| Throughput (req/s) | Requests completed per second | No (higher) |
| Token throughput (tok/s) | Output tokens generated per second | No (higher) |
| TTFT (Time to First Token) | Latency before first token — prefill speed | Yes |
| TPOT (Time Per Output Token) | Average inter-token latency — decode speed | Yes |
| P50 latency | Median end-to-end latency | Yes |
| P90 latency | 90th percentile latency | Yes |
| P99 latency | 99th percentile latency (tail) | Yes |

## vllm bench serve Command

```bash
vllm bench serve \
    --host 127.0.0.1 \
    --port 8000 \
    --backend openai-chat \
    --model <model_name> \
    --tokenizer <model_path> \
    --dataset-name random \
    --endpoint /v1/chat/completions \
    --ignore-eos \
    --trust-remote-code \
    --random-input-len <INPUT_LEN> \
    --random-output-len <OUTPUT_LEN> \
    --num-prompts <NUM_PROMPTS>
```

**Note on --model:** Must match the model name registered in the running vllm server.
Query with: `curl -s http://localhost:8000/v1/models`

## Customization

Users may override:
- **Custom profiles:** different input/output lengths or prompt counts
- **Custom args:** e.g. `--request-rate` for rate-limited benchmarks
- **Subset:** run only specific profiles instead of all 5

## Timeout

Each profile: 600s (10 minutes). If exceeded, kill and report partial.
