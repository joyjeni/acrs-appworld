# O4 — PID-controlled provenance pruning

The registered title of the thesis names this objective "PID-Controlled Provenance Pruning". The first
AppWorld implementation (`HistoryGovernor`) compressed text against a floor on regex-matched lines. That made it a
fidelity-controlled compressor, not provenance pruning, and its fidelity score was close to 1 by construction: the
dev pilot recorded 0.9999 against 0.47 for the baseline's last-six-turns window, because the governor keeps exactly
the lines the score counts. This document describes the redesign that makes the implementation match the title,
gives O4 an outcome it can be judged on with AppWorld data, and connects it to the other four components.

## Gap

| Line of work | What it does | What it cannot do |
| --- | --- | --- |
| Prompt compression: LLMLingua (Jiang et al., EMNLP 2023), LongLLMLingua (ACL 2024), LLMLingua-2 (Pan et al., Findings of ACL 2024) | drops tokens judged unimportant from the text | know that a printed value is still held by a program variable |
| Agent history processing: SWE-agent (Yang et al., NeurIPS 2024) | keeps the last few observations, collapses older ones | tell an observation that can be recovered from one that cannot |
| Code actions in a persistent interpreter: CodeAct (Wang et al., ICML 2024), AppWorld (Trivedi et al., ACL 2024) | tool results live in interpreter variables across steps | — the prompt still carries raw observations and says nothing about what the session holds |
| Retrieval compression: RECOMP (Xu et al., ICLR 2024) | compresses retrieved documents | reason about interpreter state |
| Long-context use: Liu et al., TACL 2024 | shows models use information in the middle of long contexts poorly | — motivates keeping the context short |

**Gap.** In a code-acting agent, most of an observation is the printed value of a variable that stays alive in the
interpreter. Dropping it loses nothing if the agent is told where the value lives. That is its where-provenance in
the sense of Buneman, Khanna and Tan (ICDT 2001): the variable, the API call that produced it, and its structure.
A value that was printed but never stored is lost when it is dropped. No existing context-management method makes
this distinction. None shares the resulting provenance with the agent's retrieval, routing, verification and
governance components.

## Objective (for the proposal)

> To design and evaluate a PID-controlled provenance-pruning controller for code-acting LLM agents. The controller
> replaces stale tool observations with where-provenance records of the interpreter state (variable, source API
> call, structure) and a one-line outline of every earlier step. A PID loop holds the history at the size of the
> baseline's recent-turn window, so the extra information costs no extra context. The controller shares its
> provenance store with retrieval (O1), routing (O2), consensus verification (O3) and governance (G1).

**RQ4.** When a code-acting agent's tool results persist in interpreter variables, does pruning observations by
provenance reduce the context the agent reads without lowering task completion? Do retrieval, routing,
verification and governance gain from the shared provenance?

## Design (`paim.appworld.provenance`)

1. **Interpreter state.** `InterpreterState` reads AppWorld's REPL namespace (`world.shell.user_ns`). It reports only
   the agent's own variables: names present at the start (imports, `apis`, AppWorld's preamble), private names and
   modules are excluded.
2. **Provenance.** After each committed step, `bindings()` parses the code and records each name the step binds,
   with the API calls that produced it and the step number. It covers assignments, `x += ...`, `x = f(x)`, loop
   targets and `with`/walrus targets; function and class bodies and comprehension variables are excluded.
3. **Recoverability.** A step's observation is recoverable when two conditions hold. First, no `print()` argument
   calls an API directly. Second, every name a `print()` reads is a live variable still bound to the same object
   it had right after that step. A loop variable stands for the collection it iterates over.
4. **Pruning** (`ProvenanceGovernor`). The last three turns stay verbatim. Every older turn becomes one line of a step
   summary, `[step] the calls it made -> result`. The result is, in order of preference:
   - its error message;
   - a short output in full;
   - `output held in contacts`, when a live, unchanged variable still holds the output;
   - otherwise the output cut to `_chars` characters, with a note that no variable holds it.

   The baseline's window instead drops everything older than six turns.
5. **Ledger.** Each step, the variables are listed after the latest output. For each one the ledger shows the type,
   the size, the dict keys or the keys of the first item, and the source API call and step. It never shows the
   text of a string, so tokens and passwords stay out of the prompt.
6. **PID control.** `_chars` is the control variable. The setpoint is the size the baseline's last-six-turns window
   would have at this step, capped by `o4_budget_tokens` (4,000). The error is `(setpoint − used) / setpoint`, with
   proportional, anti-windup integral and derivative terms.
   - O4 tracks the baseline's context cost.
   - What pointers save (a listing is not repeated once a variable holds it) is spent on keeping more of the past.
7. **Recoverability (measured, not enforced).** φ is the share of past non-trivial observations still visible or
   held by a live variable. It is computed from the interpreter state, not from the pruner's own text matches. For
   the window, φ counts only what the window still shows.

## Integration with the other components

| Component | What it gets from O4 | Where |
| --- | --- | --- |
| O1 retrieval | apps whose login result is held by a live variable get no login docs (they are already authenticated) | `SessionApiRetriever.retrieve(..., authenticated=)` |
| O2 routing | a step whose context lost unrecoverable information counts as risky, so it leans to the large model | `TierRouter.context(..., lossy=)` |
| O3 consensus | trial runs leave no variables behind: AppWorld restores the app databases but not the Python namespace, so the losing candidates' bindings are rolled back. The block O3 commits is checked by G1 again if it differs from the greedy block | `OutcomeConsensus.run(..., state=)`, agent loop |
| G1 governance | `R-AW-PROV`: an id, email or phone number written as a literal by a state-changing call must appear in the task or in something the agent was shown, including observations already pruned | `AppWorldPolicy.check(code, ledger)` |

`no_o4` removes O4 and every one of these links, because each of them reads O4's provenance store.

## Hypotheses and evaluation

Per step, O4 costs about what the window costs, because the PID tracks the window's size. Its claimed effect is
therefore on how the agent uses its steps. An agent that can see what it already did, and where the values are,
should not log in again, fetch the same listing again or guess a key. That should save steps, and so tokens, per
task.

| | Hypothesis | Test |
| --- | --- | --- |
| H4.1 (primary) | fewer prompt tokens per task than with the last-six-turns window | Wilcoxon signed-rank, paired over dev tasks: `o4_only` vs `baseline`, and `full` vs `no_o4` |
| H4.2 | task goal completion does not fall | McNemar exact test and confidence interval of the difference, same pairs |
| H4.3 (mechanism) | fewer steps per task, repeated read calls, logins, `KeyError`/`NameError` steps and context-limit drops | counts per task, same pairs |
| H4.4 (offline, no model) | for the same recorded actions, a per-step history no larger than the window's, higher recoverability, and far smaller than the first O4's | `experiments/replay_o4.py` on the baseline's AppWorld logs (CPU) |
| H4.5 (integration, exploratory) | `R-AW-PROV` blocks guessed values without blocking observed ones | denials and what followed each denial |

Configs in `experiments/appworld_run.py`:

- `baseline`;
- `o4_only` (the baseline plus O4: the cheapest clean test);
- `full` and `no_o4`;
- `o4_fidelity_only` and `full_o4_fidelity`: the first O4 design, for a comparison inside the chapter.

Run statistics: `promptTokens`, `meanHistoryTokens`, `o4` (pointers, condensed errors, cuts), `refetchCalls`,
`loginCalls`, `provDenials`. The notebook prints the paired tests.

`experiments/replay_o4.py` reads a recorded run's own AppWorld logs and executes the same code again in fresh worlds
(CPU only). It reports the per-step history each policy would show for those actions: the window, the first O4, and
provenance pruning. This tests H4.4 before any GPU time is spent.

## Threats to validity

- **In-place mutation.** Rebinding is tracked, in-place mutation is not. If a variable is changed in place
  (`items.append(...)`), a pointer refers to its current value, not the printed one. The same applies to O3's
  rollback of trial bindings.
- **Object identity.** Python can reuse the identity of a freed object, so in rare cases a rebound variable can look
  unchanged.
- **Remote worlds.** For a remote AppWorld environment there is no namespace. O4 then falls back to what the code
  alone says, and no observation counts as recoverable.
- **Scope of R-AW-PROV.** The rule checks literals only. A guessed value computed in code is not caught, and a short
  integer id (under three digits) is not checked.
- **What the ledger adds.** The ledger gives the agent information the baseline never had: the keys of values it
  did not print. Part of any gain may come from this rather than from pruning. A ledger-only or pruning-only run
  would separate the two if a reviewer asks.
