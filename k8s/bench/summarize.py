"""Сводит JSON-результаты vllm bench serve в markdown-таблицу для README."""
import json
import sys
from pathlib import Path

COLUMNS = [
    ("Сценарий", lambda n, r: n),
    ("Вход/выход, ток.", lambda n, r: f"{round(r['total_input_tokens'] / r['completed'])}/{round(r['total_output_tokens'] / r['completed'])}"),
    ("Клиентов", lambda n, r: r.get("max_concurrency") or 1),
    ("Запросов", lambda n, r: r["completed"]),
    ("TTFT p50, мс", lambda n, r: f"{r['median_ttft_ms']:.0f}"),
    ("TTFT p95, мс", lambda n, r: f"{r['p95_ttft_ms']:.0f}"),
    ("TPOT p50, мс", lambda n, r: f"{r['median_tpot_ms']:.0f}"),
    ("Latency p50, с", lambda n, r: f"{r['median_e2el_ms'] / 1000:.1f}"),
    ("Latency p95, с", lambda n, r: f"{r['p95_e2el_ms'] / 1000:.1f}"),
    ("Throughput, req/s", lambda n, r: f"{r['request_throughput']:.2f}"),
    ("Output, tok/s", lambda n, r: f"{r['output_throughput']:.1f}"),
    ("Total, tok/s", lambda n, r: f"{r['total_token_throughput']:.1f}"),
]

rows = []
for path in sorted(Path(sys.argv[1]).glob("*.json"), key=lambda p: (not p.stem.startswith("short"), p.stem)):
    result = json.loads(path.read_text())
    rows.append([str(fn(path.stem, result)) for _, fn in COLUMNS])

print("| " + " | ".join(c for c, _ in COLUMNS) + " |")
print("|" + "---|" * len(COLUMNS))
for row in rows:
    print("| " + " | ".join(row) + " |")
