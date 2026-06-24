# DATA_PLAN — a stronger terminal-agent SFT mixture (Terminus-2 + SWE)

**Thesis (from the precedent's verdict).** The openthoughts-agent REPORT.md gate
eval was unambiguous: after 3 epochs on the 15.2k `OpenThoughts-Agent-v1-SFT`
traces, the Qwen3-8B policy solved **0/70** TB-dev tasks (only 5/70 showed any
continuous-score spread). The conclusion: *"more/better SFT data is the dominant
lever"* and the policy is **below the completion floor**, not merely un-tuned. The
current leaderboard leader is a **Qwen3.5-27B with a SWE-heavy SFT blend** — i.e.
both *more capacity* and *more software-engineering trajectory data*. This plan
targets the data lever: a much larger, SWE-weighted terminal-agent SFT mixture
that is **drop-in compatible with the ported encoder** (`training/agent_sft.py`).

## Why this is cheap to wire

The ported `agent_data/agent_traces.py` reads HF parquet shards directly with
pyarrow and yields `{"messages": [{"role","content"}, ...]}`; the encoder masks
loss to assistant turns. **Any dataset whose rows expose either a `conversations`
list (role/content) OR a `messages` list slots in with at most a column-rename.**
The decisive discovery below: a large family of datasets is in the **exact
`conversations` + Terminus-2 schema** of the precedent's own SFT set — same
`agent: terminus-2`, same 9 metadata columns — so they need **zero** code change.

## The mixture (proposed proportions, ~per-epoch token budget)

Weights are *rough sampling proportions* (by usable trace, capped per source so no
single corpus dominates). Start ~120–180k traces/epoch (≈8–12× the precedent's
15.2k), SWE-weighted ~55%, in keeping with the leader's SWE-heavy blend.

| # | bucket | datasets (HF id) | format | weight | rationale |
|---|---|---|---|---|---|
| 1 | **Terminus-2 terminal agent (in-domain core)** | `open-thoughts/OpenThoughts-Agent-v1-SFT` (15.2k) | `conversations`/terminus-2 | 12% | the proven base; exact eval distribution |
| 2 | **Terminus-2 SWE-agent traces (NEW)** | `DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes-traces-terminus-2` (12.0k) | `conversations`/terminus-2 | 18% | **SWE trajectories already in the Terminus-2 action format** — zero-rename drop-in; the single highest-value add |
| 3 | **Terminus-2 shell/unix Q&A (NEW)** | `mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2` (9.99k), `…stackexchange-superuser-…-terminus-2` (9.98k), `…staqc-…-terminus-2`, `…swesmith_with_plain_docker-…-terminus-2`, `…stackexchange-codereview/-tor-…-terminus-2` | `conversations`/terminus-2 | 22% | broad terminal/unix competence in the exact harness format; a whole OpenThoughts-pipeline corpus the precedent never used |
| 4 | **SWE-bench-style repo-fix trajectories (NEW)** | `SWE-bench/SWE-smith-trajectories` (76.0k), `nebius/SWE-agent-trajectories` (80.0k) | `messages` (system/user/assistant + tool) | 22% | large, high-quality SWE-agent SFT corpora (the leader's lever); column-map `messages`→encoder |
| 5 | **OpenHands / R2E SWE SFT (NEW)** | `R2E-Gym/R2EGym-SFT-Trajectories` (3.23k), `SWE-Gym/OpenHands-SFT-Trajectories` (491) | `messages` | 6% | small, *successful-only* trajectories (high signal-per-row) for repo navigation/patching |
| 6 | **Tool-calling / function-calling (format breadth)** | `HuggingFaceTB/smoltalk2` SFT splits `smolagents_toolcalling_traces_think` (9.08k), `hermes_function_calling_v1_no_think` (8.96k), `xlam_traces_no_think` (59.96k); `nvidia/Nemotron-Post-Training-Dataset-v1` split `tool_calling` (310k, subsample) | `messages` + tool schema | 14% | keeps structured-action / JSON discipline sharp; smoltalk2's `xlam_traces` is the un-gated reformat of Salesforce/xlam |
| 7 | **Code SFT (coding competence)** | `TIGER-Lab/AceCode-89K` (87.1k, filter by `pass_rate`, subsample) | `context_messages`/`inferences` | 6% | raw code-writing ability under the terminal agent |

Caps and curriculum:
- **Cap per source** (e.g. ≤25k traces) so the long tails (Nemotron 310k, xlam
  60k, AceCode 87k) don't swamp the in-domain Terminus-2 buckets (1–3).
- **Quality filter where a signal exists:** SWE-smith/R2E/OpenHands ship a
  `resolved`/success flag — prefer resolved trajectories; AceCode has `pass_rate`;
  Nemotron `tool_calling` has a `reasoning` flag.
- **Length:** these are multi-turn agent episodes — keep `MAX_SEQ_LEN 8192` (the
  fitted 8B envelope) and let `encode_agent_conversation` drop episodes that
  truncate before any assistant token (it already logs `window_full`).
- **Reject-sampled self-traces (the precedent's own "next step"):** once a first
  blend is trained, fold in the agent's *own* grader-passing rollouts (the eval
  harness already produces these) as bucket 8 — the cheapest way to lift all-zero
  TB tasks into the trainable band.

## The 4 named NEW SWE/terminal datasets the precedent + marin LACK

The precedent used only `OpenThoughts-Agent-v1-SFT`; marin's
`instruction_datasets.py` has tool/code slices (smoltalk2 tool, Nemotron
tool_calling, xlam, AceCode) but **no SWE/terminal-trajectory data at all**. These
four (plus the broader families) fill exactly that gap; all verified accessible
(ungated, resolve via `huggingface_hub`, `2026-06-23`):

1. **`DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes-traces-terminus-2`**
   — 12,015 rows, 112 MB, ungated. **Schema identical to the precedent's SFT set**
   (`conversations`, `agent=terminus-2`, same 9 metadata cols). SWE-agent
   trajectories (Nebius/neulab) replayed through the **Terminus-2 sandbox harness**
   → drop-in with the existing `agent_traces.py`/encoder, **no code change**. The
   single best add.
2. **`mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2`** (9,987
   rows) and its siblings **`…-superuser-…`** (9,983), **`…-staqc-…`**,
   **`…-swesmith_with_plain_docker-…`**, **`…-stackexchange-codereview/-tor-…`** —
   an entire `*-sandboxes-traces-terminus-2` corpus from the OpenThoughts
   data-generation pipeline, all in the same Terminus-2 `conversations` schema.
   Broad unix/shell/SWE competence the precedent never touched. (Dozens of repos
   under `mlfoundations-dev`; the unix/superuser/staqc/swesmith ones are the
   high-value, non-eval-set members.)
3. **`SWE-bench/SWE-smith-trajectories`** — 76,002 rows, 4.2 GB, ungated.
   `messages` format (system + user/assistant + tool calls) over `instance_id`,
   `resolved`, `patch`. The canonical large SWE-agent SFT corpus; the leader-style
   SWE lever. Needs a `messages`→encoder column path (trivial) and a
   `resolved==True` filter for the high-signal subset.
4. **`nebius/SWE-agent-trajectories`** — 80,036 rows, 1.1 GB, ungated. `messages`
   SWE-agent SFT trajectories; pairs with `nebius/SWE-rebench-V2` (a large
   verified SWE task set) if you want to *generate* fresh traces in your own
   Terminus-2 harness later.

Runners-up worth a look (also new vs. precedent+marin): `R2E-Gym/R2EGym-SFT-
Trajectories` (3.2k, success-only), `SWE-Gym/OpenHands-SFT-Trajectories` (491,
success-only), `nvidia/SWE-Hero-openhands-trajectories` (2.1k),
`jiacheng-ye/nl2bash` (~12k NL→shell pairs, classic terminal grounding),
`harborframework/terminal-bench-2.0` (a newer/larger Terminal-Bench — candidate to
*expand the eval set* beyond the 70 TB-dev tasks).

## Integration sketch (no code shipped here — just the shape)

- Buckets 1–3 (Terminus-2 `conversations`): point `agent_data/agent_traces.py` at
  each `repo_id` (its loader already does `snapshot_download(... allow_patterns=
  ["data/*.parquet"]) → pyarrow → {"messages": conv}`); only `DATASET_ID`/revision
  differ. A thin multi-source wrapper that round-robins/weights several `repo_id`s
  is the only new code.
- Buckets 4–7 (`messages`/`context_messages`): one adapter that renames the column
  to `messages` and maps roles (`system`→folded into first user, to match the
  no-system-turn SFT layout the agent expects; or keep `system` and let the encoder
  mask it — it already masks every non-assistant role). SWE-smith `messages` carry
  tool-call turns; the encoder masks them as context, which is correct.
- Keep the **assistant-turn loss mask** and the **`<|im_end|>` stop-token
  training** exactly as ported — they are what makes the rollout harness's stop
  condition work.

## Expected effect

This raises in-domain SWE/terminal trace volume from 15.2k to ~150k with a
SWE-heavy tilt, in the *exact* action format the eval harness scores — directly
attacking the "below the completion floor" finding. The Terminus-2 buckets (1–3)
need no new encoder code; the SWE `messages` buckets (4–7) need one small adapter.
After a first blend, close the loop with grader-passing self-traces.
