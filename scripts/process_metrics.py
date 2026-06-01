"""Extract evaluation metadata and compile p50/p95 telemetry.

Parses report.json from pytest-json-report, computes latency percentiles,
tracks overall execution pass rates, and appends a summarized trend entry
to a running history log file for dashboard rendering.
"""

import json
import math
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "report.json"
HISTORY_PATH = REPO_ROOT / "history.json"

def calculate_percentile(durations, percentile_target):
    """Calculate specific percentile using linear interpolation."""
    if not durations:
        return 0.0
    sorted_data = sorted(durations)
    index = (len(sorted_data) - 1) * percentile_target
    floor_idx = math.floor(index)
    ceil_idx = math.ceil(index)
    if floor_idx == ceil_idx:
        return sorted_data[int(index)]
    return sorted_data[floor_idx] * (ceil_idx - index) + sorted_data[ceil_idx] * (index - floor_idx)

def main():
    if not REPORT_PATH.exists():
        print(f"❌ Error: {REPORT_PATH} not found. Cannot compile pipeline metrics.")
        return

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Gather test durations and outcomes
    durations = []
    total_tests = 0
    passed_tests = 0

    for test_run in report_data.get("tests", []):
        total_tests += 1
        if test_run.get("outcome") == "passed":
            passed_tests += 1
        
        # Pull duration info from setup, call, and teardown stages
        call_stage = test_run.get("call", {})
        duration = call_stage.get("duration", 0.0)
        if duration > 0.0:
            durations.append(duration)

    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
    p50_latency = calculate_percentile(durations, 0.50)
    p95_latency = calculate_percentile(durations, 0.95)
    
    # Cost Estimation logic for Gemini 2.5 Flash free tier
    # Free-tier cost is exactly 0.0, but we trace it structurally so it works if upgraded
    estimated_cost = 0.0 

    # Compile the telemetry package for this run
    summary_metrics = {
        "commit_sha": os.getenv("GITHUB_SHA", "local-run")[:7],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_cases": total_tests,
        "pass_rate_percent": round(pass_rate, 2),
        "p50_latency_sec": round(p50_latency, 3),
        "p95_latency_sec": round(p95_latency, 3),
        "estimated_cost_usd": estimated_cost
    }

    # Append to running historical record ledger
    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []

    history.append(summary_metrics)

    # Keep only the last 50 entries to keep dashboard assets small
    history = history[-50:]

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n📊 --- PIPELINE TELEMETRY SUMMARY ---")
    print(f"✅ Pass Rate:       {summary_metrics['pass_rate_percent']}%")
    print(f"⏱️ P50 Latency:     {summary_metrics['p50_latency_sec']} seconds")
    print(f"⚡ P95 Latency:     {summary_metrics['p95_latency_sec']} seconds")
    print(f"💳 Estimated Cost:  ${summary_metrics['estimated_cost_usd']} USD")
    print("-------------------------------------\n")

if __name__ == "__main__":
    main()