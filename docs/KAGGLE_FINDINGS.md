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

## The 20-task dev pilot of this repository (November 2026) and the two changes it led to

Kaggle 2 x T4, vLLM, Qwen3-8B-AWQ (large tier) and Qwen3-4B-AWQ (small tier), `--context-limit 16384`, the first
20 `dev` tasks. Only `baseline` and `full` ran: the 1.6 h budget (`BUDGET_H`) stopped the leave-one-out ablations.

| | baseline | full |
| --- | --- | --- |
| TGC (tasks passed) | 0/20 (Wilson 95% CI 0.00-0.16) | 0/20 (0.00-0.16) |
| tokens | 1.55 M | 7.54 M |
| wall time | 69 min | 252 min |

Both configs are at the floor, so this pilot says nothing about any objective. It did show two problems.

1. **O3 hid errors from the agent.** In `full`, O3 ran on 546 steps: 31 certified, 10 abstained because the
   candidates disagreed, and 505 abstained because every candidate raised. On those 505 steps the agent was told
   only "Not executed: all candidates raised errors" and the candidates' call signatures, never the error message;
   the baseline sees the message on every failing step. A login is a POST, so every login step also went through
   the vote (three candidate runs and the commit), and candidates that wrote the credentials differently counted as
   disagreeing.
   *Change:* when every candidate raises, the agent gets the greedy block's own output (the same message it would
   get without O3) and is told that nothing was committed; the error counts as an error step and is sampled into
   `error_samples`. `login` is exempt from the vote (`OutcomeConsensus.exempt_apis`) and left out of the
   signatures candidates are compared on; exempt steps are counted (`consensus.exempt_steps`).
2. **The baseline is too weak to compare against.** The most frequent error lines in the baseline were syntax
   errors (11), `KeyError: 'email'` (10) and Venmo "unauthorized" / missing access token (8): mistakes about how
   the APIs fit together. The AppWorld paper's full-code baseline puts three solved train tasks in the prompt
   (StonyBrookNLP/appworld at v0.1.3.post1: `experiments/configs/full_code_refl_*.jsonnet`, tasks 82e2fac_1,
   29caf6f_1, d0b1f43_1; `experiments/code/plain/agents/full_code_reflexion_agent.py` shows each task's compiled
   ground-truth solution with comments removed).
   *Change:* `--few-shot K` (notebook `FEW_SHOT`, default 3) does the same for every config, from the downloaded
   data at run time (`paim.appworld.demos`): the authors' demonstration tasks first, then one task per scenario
   favouring apps not yet shown. Train split only, and never a task listed in another split. Solutions that use
   AppWorld-internal names (`requester`, `main_user`, `public_data`, ...) are skipped. The experiment name records K
   (`acrs_<config>_fs3_s0_dev`), so runs with and without examples never share an episode log.
   `experiments/check_demos.py` checks the selection before any GPU time; it prints task ids only, since AppWorld
   asks that its task content is not published.

Two measurements change with these fixes, so O3 numbers from this pilot are not comparable with later runs:
`steps` now counts every model turn (blocked and abstained turns included; before, a run that ended on such
turns undercounted), and steps on which every O3 candidate raised count as error steps.

