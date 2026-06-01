# llm-eval-cicd

## Executive Summary

llm-eval-cicd is a lightweight, Python-based evaluation harness for validating LLM-driven responses against a golden dataset. It demonstrates a free-tier evaluation stack that uses the `google-generativeai` (Gemini) SDK and the `deepeval` test harness to run automated metric-driven checks (hallucination, relevancy, faithfulness) against predefined question-answer-context triples.

This repository is intended as a minimal evaluation CI/CD reference for teams that need to validate LLM outputs in reproducible pipelines, or as a foundation for building enterprise-grade evaluation automation.

## Key Features

- Golden dataset-driven evaluation using a simple JSON schema (`tests/dataset.json`).
- Pluggable model adapter that allows DeepEval to use Google Gemini (`gemini-1.5-flash`) instead of OpenAI.
- Metric assertions via `deepeval` metrics: HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric.
- Dataset generation script that can synthesize customer-support Q/A/context trios using Gemini.
- PyTest-based harness for running evaluation cases and integrating into CI.

## Architecture Overview

The repository is intentionally small; the core components detected are:

- `tests/test_llm.py` — evaluation harness, metrics wiring, Gemini judge adapter, and a mock pipeline suitable for local runs.
- `scripts/generate_dataset.py` — synthetic dataset generator using Gemini.
- `tests/dataset.json` — golden dataset JSON array used to parameterize tests.

Mermaid architecture diagram:

```mermaid
graph TD
  A[Developer / CI Runner] -->|run pytest| B[tests/test_llm.py]
  B --> C[DeepEval metrics]
  B --> D[Gemini Judge Adapter]
  D --> E[Google Gemini API (gemini-1.5-flash)]
  F[tests/dataset.json] --> B
  G[scripts/generate_dataset.py] --> E
  G --> F
```

## Technology Stack

| Area | Technologies (observed in repo) |
|---|---|
| Language | Python 3.x (files target Python environment) |
| AI/LLM SDK | google-generativeai (Gemini) |
| Evaluation | deepeval (DeepEval metrics) |
| Testing | pytest |
| Scripting | Native Python scripts in `scripts/` |

Not detected in repository: frontend frameworks, Dockerfiles, Kubernetes manifests, database schemas, or server code beyond test harness and generator scripts.

## Repository Structure

| Path | Purpose |
|---|---|
| `tests/test_llm.py` | PyTest evaluation harness that loads `tests/dataset.json`, configures a Gemini-backed DeepEval judge, and runs metric assertions. |
| `tests/dataset.json` | Golden dataset: an array of objects with keys `input`, `expected_output`, and `context`. |
| `scripts/generate_dataset.py` | Generator script that can synthesize dataset items using the Gemini API with batching and rate limiting. |
| `README.md` | This file. |

## System Design

Request flow (evaluation run):

1. PyTest runner invokes `tests/test_llm.py`.
2. `load_golden_dataset()` reads `tests/dataset.json` and yields test cases.
3. For each test case, `run_your_llm_pipeline()` (mock or real Gemini call) produces `actual_output`.
4. `LLMTestCase` instances are built and passed to `deepeval.assert_test()` evaluated against configured metrics.
5. Metrics may call the Gemini judge adapter (configured via `GEMINI_API_KEY`) to perform evaluation-related model queries.

Data flow:

- Golden data lives in `tests/dataset.json` and is immutable to the test runner (overwritten only by the generator script).
- Results are produced by pytest; the repository contains no built-in result storage beyond local file writes performed by scripts.

Service interactions:

- Local runner ↔ Gemini API (via `google-generativeai`) — used by both generator and judge adapter.

State management:

- Minimal; tests are stateless per-run. The generator writes `tests/dataset.json` when invoked.

## Prerequisites

- Python 3.10+ recommended for parity with CI examples and modern package support.
- A Gemini API key (see Environment Variables section).

Not detected in repository: `requirements.txt` or `pyproject.toml`. The repository imports the following third-party packages in code and therefore they must be installed in the environment before running tests or scripts:

- deepeval
- google-generativeai
- pytest
- pytest-json-report (recommended for --json-report CLI flag in CI)

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `GEMINI_API_KEY` | Yes (used in `tests/test_llm.py`) | API key for Google Gemini / google-generativeai. Required for generator and judge integrations. |
| `GOOGLE_API_KEY` | Optional (fallback in `scripts/generate_dataset.py`) | Alternate name for Gemini key — generator script checks this variable as a fallback. |

If these variables are not set, code paths that depend on Gemini will raise a `ValueError` or `RuntimeError` (see `tests/test_llm.py` and `scripts/generate_dataset.py`).

## Installation

Install the observed runtime dependencies into a virtual environment. The repository lacks a `requirements.txt` or `pyproject.toml`, so install the packages listed in the Environment section.

Example (local development):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install deepeval google-generativeai pytest pytest-json-report
```

## Running the Project

Run the unit/evaluation harness locally:

```bash
export GEMINI_API_KEY="<your_key>"
pytest tests/test_llm.py
```

Generate or refresh the golden dataset (this will call Gemini; ensure API key is present):

```bash
python scripts/generate_dataset.py
```

Notes:

- The generator implements batching and a 5-second sleep between batches to respect Gemini free-tier RPM limits.
- The test harness includes a per-case `time.sleep()` to avoid hitting rate limits during metric calls.

## Available Scripts

| Command | Purpose |
|---|---|
| `python scripts/generate_dataset.py` | Generate synthetic golden dataset (writes `tests/dataset.json`). |
| `pytest tests/test_llm.py` | Run the evaluation harness locally. |

## API Documentation

No web API endpoints were detected in the repository.

## Database Design

Not detected in repository.

## Authentication & Authorization

Authentication to the external LLM provider is done via environment variable `GEMINI_API_KEY` (and `GOOGLE_API_KEY` as a fallback in the generator). No other auth flows were detected.

## CI/CD Pipeline

No CI workflow files were detected in `.github/workflows/` in the repository. Example CI usage (user-provided) ran pytest with `--json-report`; note that `--json-report` is provided by the `pytest-json-report` plugin, which is not detected in the repository dependencies.

Suggested CI step to install runtime dependencies (example):

```yaml
- name: Install dependencies
  run: |
    pip install deepeval pytest google-generativeai pytest-json-report
```

## Monitoring & Observability

Not detected in repository: monitoring agents, logging backends, or observability configurations.

## Security Considerations

| Area | Notes |
|---|---|
| Secrets management | Gemini API key must be provisioned via CI secrets (e.g., GitHub Secrets) and never committed to source. |
| Input validation | `tests/test_llm.py` and `scripts/generate_dataset.py` perform JSON validation on dataset payloads. |
| Network calls | Generator and judge adapter call external APIs; consider VPC egress controls in corporate environments. |

No automated secret scanning or hardening configuration files were detected in the repository.

## Performance Optimizations

- The generator uses batching and sleep delays to respect API rate limits. No additional performance optimizations are present.

## Testing Strategy

- Tests are driven by `pytest` and parameterized by `tests/dataset.json`.
- DeepEval is used to assert metrics per test-case.

## Troubleshooting Guide

- If tests fail with authentication errors: ensure `GEMINI_API_KEY` is set in the environment.
- If generation fails with rate-limit errors: ensure the generator's batching and `time.sleep()` are respected; consider increasing the delay.
- If `--json-report` fails in CI: ensure `pytest-json-report` is installed in the CI environment.

## Deployment Guide

No deployment artifacts or service manifests were detected. This repository contains test harnesses and scripts intended to be run from developer machines or CI runners.

## Contributing Guidelines

Not detected in repository: CONTRIBUTING.md. For contributions, submit PRs against `main`, include unit tests, and avoid adding secrets.

## Coding Standards

The codebase is small and Pythonic. No formal linter or formatter configs (e.g., `pyproject.toml`, `.flake8`) were detected.

## Future Improvements

1. Add a `requirements.txt` or `pyproject.toml` to lock dependencies for reproducible runs.
2. Add CI workflow YAML to the repository (example user-provided snippet suggests GitHub Actions).
3. Add a `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for open-source governance.
4. Add structured result storage and artifact upload (e.g., S3, Artifactory) for evaluation runs.
5. Add monitoring and CI integration tests to validate the rate-limiting behavior automatically.

## Self-review & Missing Sections

The README was generated from the repository contents. The following items were not detected and therefore could not be documented with code-backed detail:

- Dockerfiles: Not detected in repository.
- Kubernetes manifests / Helm charts: Not detected in repository.
- CI workflow files under `.github/workflows/`: Not detected in repository.
- `requirements.txt` or `pyproject.toml`: Not detected in repository.
- Formal contributing or governance documents: Not detected in repository.
- Web services, API routes, database schemas, or server application layers: Not detected in repository.

Verify accuracy: this README references only files and imports present in the codebase (`tests/test_llm.py`, `scripts/generate_dataset.py`, `tests/dataset.json`) and recommends adding missing infra/dependency files for enterprise adopters.

## License

Not detected in repository.
# llm-eval-cicd