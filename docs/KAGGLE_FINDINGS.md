# Harness findings from the Kaggle runs (8 November 2026)

A separate ReAct harness for the same research (Qwen3-8B fp16 on Kaggle 2 x T4, AppWorld 0.1.3.post1, `dev`)
passed a Phase 0 gate and ran a 5-task pilot. These are the harness problems it hit, checked against this
repository. Everything ported here is part of the shared harness: the baseline and every ablation get it alike.

| Finding on Kaggle | Status in this repository before | Change |
| --- | --- | --- |
| AppWorld 0.1.3 does not install on the kernel's Python 3.13; it pins pydantic 1 / uvicorn 0.27 | already handled: separate Python 3.11 venv for AppWorld | none |
| AppWorld freezes time in-process (freezegun), so `time.time()` and `time.perf_counter()` return a fixed instant: every latency read 0.0 s | `ChatClient` and the agent timed with `time.time()`, so per-call seconds and episode seconds were 0 inside workers | `paim.common.clock` keeps the real clock; client, agent and runner use it |
| The shell returns only printed output. A bare expression (`passwords`) comes back as "Execution successful."; Qwen3-8B then invented values ("the available apps are Spotify, YouTube and Netflix") | the prompt asks for `print(...)`, nothing enforced it | a trailing bare expression is printed (`show_last_expression`); an API call that printed nothing gets a one-line hint (`with_silence_hint`) |
| Prompts grow past the server's context; the request fails and the whole episode is lost | baseline keeps the last 6 turns (each output up to 6,000 characters); no hard limit, so long outputs can exceed vLLM's `--max-model-len` (10k/16k on T4) | `AgentConfig.context_limit_tokens` (runner `--context-limit`, notebook passes `LEN`): oldest turns are dropped first, the latest turn is always kept; drops are counted (`window_dropped`) |
| A server error (HTTP 400/500) was reported without its message | `HTTPError` retried and re-raised without the body | the error carries the server's message (e.g. maximum context length, CUDA out of memory) |
| Kaggle ends a session at 12 h, and the weekly quota is 30 h | per-config time budget (`BUDGET_H`) | `ACRS_DEADLINE` stops workers between tasks; `--resume` skips tasks already in the episode log; train/dev episodes record AppWorld's own per-task result, so tasks whose app databases are gone (a new session) are scored from the record instead of as failures; partial runs report TGC/SGC over attempted tasks (`tasksAttempted`, `stoppedByDeadline`) |
| `appworld evaluate` on a partial split fails on missing task outputs | already handled: `evaluate_tasks` on the scorable subset | the recomputed aggregate now covers attempted tasks only, and keeps the official subset aggregate alongside |
| Transformers on a T4 (no flash attention): SDPA with grouped-query attention falls back to the math kernel and runs out of memory at a few thousand tokens | not affected: vLLM serves the models (smoke run validated on Kaggle, 5 Nov) | none; if vLLM ever fails on the T4, expand the KV heads before SDPA so the memory-efficient kernel runs |

## What the 5-task pilot showed (ReAct baseline, Qwen3-8B, 30 steps; a smoke test, not evidence)

0/5 tasks passed (Wilson 95% CI 0.00-0.43). Four episodes called `complete_task` with a wrong result, one hit the
step limit; mean 24.2 steps and 139k prompt tokens per task. In every task the first five steps went to listing
apps and reading API docs one at a time; the model also called APIs that do not exist (`spotify.get_access_token`)
and logged in without a username. These are the costs O1 (retrieval) and O4 (history) target, and the outcomes O3
and G1 guard. The 40-trajectory Stage 1 audit (two coding passes, Cohen's kappa) decides which of them dominate.
