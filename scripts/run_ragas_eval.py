"""
Ragas scores for the 43-question dev set (eval/eval_set_v2.json). Never touches the held-out set.

Judge LLM: Claude Haiku (JUDGE_MODEL) through the Anthropic SDK's structured outputs. Embeddings:
the local MiniLM model the pipeline already uses (src/ingest/local_embeddings.py). No Batch API:
ragas calls its LLM one prompt at a time and its metrics chain calls (faithfulness extracts
statements, then checks them), so the Message Batches API does not fit; calls run concurrently.

Inputs, per question:
  response            generated_answer from the latest run of that id: eval_v3_runB_verifier.json for
                      the 25 ids it re-ran, eval_v3_runB_full.json for the other 18 (all not_covered).
  retrieved_contexts  full parent abstracts (corpus/v3_5k_recursive.json.gz) the generator was given,
                      i.e. `sources`, in retrieval (fusion) rank order. For answered_partial that is the
                      judge-filtered subset. not_covered answers used none, so those rows carry the 6
                      retrieved-but-unused parents (`considered_sources`) and are marked as such.
  reference           hand-written reference_answer. In-scope id 23 has none, so its context metrics
                      are skipped.

Scored:
  faithfulness, answer_relevancy     answered + answered_partial rows (any scope)
  context_precision, context_recall  in-scope rows with a reference, any status

Agreement with the existing grades (eval_v3_runB_verifier_grades_manual_plus_auto.json), mapping
fixed before the run: ragas faithfulness == 1.0 -> SUPPORTED, 0.5 <= f < 1.0 -> PARTIALLY_SUPPORTED,
f < 0.5 -> UNSUPPORTED. Also reported binary: SUPPORTED vs not.

    python -m scripts.run_ragas_eval              # build dataset, score, write report
    python -m scripts.run_ragas_eval --limit 1    # smoke: first scorable id only
    python -m scripts.run_ragas_eval --report     # rebuild the report from saved scores only

Resumable: each finished (id, metric) score is appended to scores.jsonl and skipped on restart;
failed scores are logged with their error and retried on the next run.
"""
import argparse
import asyncio
import gzip
import json
import math
import os
import time
from collections import Counter

from dotenv import load_dotenv

QUESTIONS_PATH = "eval/eval_set_v2.json"
RUN_DIR = "eval/results/v3_2026-10-02"
VERIFIER_RUN = f"{RUN_DIR}/eval_v3_runB_verifier.json"
FULL_RUN = f"{RUN_DIR}/eval_v3_runB_full.json"
GRADES_PATH = f"{RUN_DIR}/eval_v3_runB_verifier_grades_manual_plus_auto.json"
CORPUS_PATH = "corpus/v3_5k_recursive.json.gz"
OUT_DIR = "eval/results/ragas"
DATASET_OUT = f"{OUT_DIR}/dataset.jsonl"
SCORES_OUT = f"{OUT_DIR}/scores.jsonl"
SUMMARY_OUT = f"{OUT_DIR}/summary.json"
REPORT_OUT = f"{OUT_DIR}/report.md"

JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_EFFORT = "medium"
JUDGE_MAX_TOKENS = 16000
CONCURRENCY = 4
ANSWERED = ("answered", "answered_partial")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
STATUS_ROWS = ("answered", "answered_partial", "not_covered")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset():
    questions = load_json(QUESTIONS_PATH)["questions"]
    parents = json.load(gzip.open(CORPUS_PATH, "rt", encoding="utf-8"))["parents"]
    verifier = {r["id"]: r for r in load_json(VERIFIER_RUN)["results"]}
    full = {r["id"]: r for r in load_json(FULL_RUN)["results"]}
    rows = []
    for q in questions:
        run, run_file = (verifier[q["id"]], VERIFIER_RUN) if q["id"] in verifier else (full[q["id"]], FULL_RUN)
        used = run["status"] in ANSWERED
        srcs = sorted(run["sources"] if used else run["considered_sources"], key=lambda s: int(s["fusion_rank"]))
        # Background pages carry no pmid; resolve every source to its parent id via the run's fused list.
        fused = [f["parent_id"] for f in run["retrieval"]["fused"]]
        ids = [next(p for p in fused if p == s.get("pmid") or parents[p].get("title") == s["title"]) for s in srcs]
        rows.append({
            "id": q["id"], "topic": q["topic"], "in_scope": q["in_scope"], "status": run["status"],
            "run_file": os.path.basename(run_file),
            "user_input": q["question"],
            "retrieved_contexts": [parents[i]["text"] for i in ids],
            "context_ids": ids,
            "contexts_kind": "used_for_generation" if used else "retrieved_not_used",
            "response": run.get("generated_answer") if used else None,
            "reference": q.get("reference_answer") or None,
        })
    return rows


def metrics_for(row):
    out = []
    if row["status"] in ANSWERED and row["response"]:
        out += ["faithfulness", "answer_relevancy"]
    if row["in_scope"] and row["reference"]:
        out += ["context_precision", "context_recall"]
    return out


class AnthropicJudge:
    """ragas InstructorBaseRagasLLM backed by messages.parse (structured outputs). No sampling
    params: Haiku 5.5 rejects non-default temperature/top_p, which ragas' own adapter sends."""

    def __init__(self, model=JUDGE_MODEL, effort=JUDGE_EFFORT):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model, self.effort = model, effort
        self.usage = Counter()

    async def agenerate(self, prompt, response_model):
        msg = await self.client.messages.parse(
            model=self.model, max_tokens=JUDGE_MAX_TOKENS, output_format=response_model,
            output_config={"effort": self.effort}, messages=[{"role": "user", "content": prompt}])
        self.usage["requests"] += 1
        self.usage["input_tokens"] += msg.usage.input_tokens
        self.usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason != "end_turn" or msg.parsed_output is None:
            raise RuntimeError(f"judge stop_reason={msg.stop_reason}, request {msg._request_id}")
        return msg.parsed_output

    def generate(self, prompt, response_model):
        return asyncio.run(self.agenerate(prompt, response_model))


class MiniLMEmbeddings:
    """ragas BaseRagasEmbedding over the pipeline's local all-MiniLM-L6-v2 (fastembed)."""

    def __init__(self):
        from src.ingest.local_embeddings import LocalSentenceTransformerEmbeddings
        self.model = LocalSentenceTransformerEmbeddings()

    def embed_text(self, text, **kwargs):
        return self.model.embed_query(text)

    def embed_texts(self, texts, **kwargs):
        return self.model.embed_documents(list(texts))

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text)

    async def aembed_texts(self, texts, **kwargs):
        return await asyncio.to_thread(self.embed_texts, texts)


def make_metrics():
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    # Registered as virtual subclasses so ragas' isinstance checks accept the adapters above.
    InstructorBaseRagasLLM.register(AnthropicJudge)
    BaseRagasEmbedding.register(MiniLMEmbeddings)
    llm, emb = AnthropicJudge(), MiniLMEmbeddings()
    return llm, {
        "faithfulness": (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=emb), ("user_input", "response")),
        "context_precision": (ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        "context_recall": (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
    }


def load_scores():
    if not os.path.exists(SCORES_OUT):
        return {}
    done = {}
    with open(SCORES_OUT, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error") is None:
                done[(rec["id"], rec["metric"])] = rec
    return done


async def score_all(rows, limit=None):
    llm, metrics = make_metrics()
    done = load_scores()
    jobs = [(r, m) for r in rows for m in metrics_for(r) if (r["id"], m) not in done]
    if limit:
        first = sorted({r["id"] for r, _ in jobs})[:limit]
        jobs = [(r, m) for r, m in jobs if r["id"] in first]
    print(f"{len(done)} scores already saved, {len(jobs)} to run")
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()

    async def one(row, name):
        metric, fields = metrics[name]
        rec = {"id": row["id"], "metric": name, "status": row["status"], "value": None, "error": None}
        async with sem:
            t0 = time.monotonic()
            try:
                rec["value"] = (await metric.ascore(**{k: row[k] for k in fields})).value
            except Exception as e:  # recorded and retried on the next run, never a silent score
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["seconds"] = round(time.monotonic() - t0, 1)
        async with lock:
            with open(SCORES_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"  id={row['id']:>2} {name:<18} {rec['value'] if rec['error'] is None else 'ERROR ' + rec['error'][:120]}")

    await asyncio.gather(*(one(r, m) for r, m in jobs))
    return dict(llm.usage)


async def explain(rows, ids):
    """Faithfulness' two steps, printed. A fresh judge call, so verdicts can differ slightly from scores.jsonl."""
    _, metrics = make_metrics()
    metric = metrics["faithfulness"][0]
    by_id = {r["id"]: r for r in rows}
    for i in ids:
        r = by_id[i]
        statements = await metric._create_statements(r["user_input"], r["response"])
        verdicts = (await metric._create_verdicts(statements, "\n".join(r["retrieved_contexts"]))).statements
        print(f"\nid {i}: {sum(v.verdict for v in verdicts)}/{len(verdicts)} statements supported")
        for v in verdicts:
            if not v.verdict:
                print(f"  - {v.statement}\n      judge: {v.reason}")


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return (round(sum(xs) / len(xs), 3), len(xs)) if xs else (None, 0)


def ragas_label(f):
    return "SUPPORTED" if f >= 1.0 - 1e-9 else "PARTIALLY_SUPPORTED" if f >= 0.5 else "UNSUPPORTED"


def summarize(rows, usage):
    scores = load_scores()
    by_id = {r["id"]: r for r in rows}
    table = {}
    for status in STATUS_ROWS + ("all",):
        table[status] = {m: mean([s["value"] for (i, mm), s in scores.items()
                                  if mm == m and (status == "all" or by_id[i]["status"] == status)]) for m in METRICS}
    lowest = {m: sorted(((s["value"], i) for (i, mm), s in scores.items() if mm == m and s["value"] is not None))[:5]
              for m in METRICS}
    grades = {g["id"]: g for g in load_json(GRADES_PATH)["results"]}
    pairs = []
    for (i, m), s in sorted(scores.items()):
        if m == "faithfulness" and i in grades and s["value"] is not None:
            g = grades[i]
            pairs.append({"id": i, "status": by_id[i]["status"], "ragas": round(s["value"], 3),
                          "ragas_label": ragas_label(s["value"]), "existing": g["overall"],
                          "grader": "manual" if g["grader"].startswith("manual") else "automated"})
    n = len(pairs)
    agree3 = sum(p["ragas_label"] == p["existing"] for p in pairs)
    agree2 = sum((p["ragas_label"] == "SUPPORTED") == (p["existing"] == "SUPPORTED") for p in pairs)
    by_grader = {k: (sum(p["ragas_label"] == p["existing"] for p in pairs if p["grader"] == k),
                     sum(p["grader"] == k for p in pairs)) for k in ("manual", "automated")}
    missing = sorted({(r["id"], m) for r in rows for m in metrics_for(r)} - set(scores))
    return {"judge_model": JUDGE_MODEL, "judge_effort": JUDGE_EFFORT, "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "table": table, "lowest": lowest, "agreement": {
                "n": n, "three_class": agree3, "binary": agree2, "by_grader": by_grader,
                "confusion": Counter(f"{p['existing']} -> {p['ragas_label']}" for p in pairs), "pairs": pairs},
            "missing_scores": missing, "usage_this_run": usage,
            "status_counts": Counter(r["status"] for r in rows)}


def fmt(cell):
    v, n = cell
    return "—" if v is None else f"{v:.3f} (n={n})"


def pct(a, n):
    return f"{a}/{n} ({100 * a / n:.0f}%)" if n else "n/a"


def write_report(s):
    a = s["agreement"]
    L = ["# Ragas scores: dev set (43 questions)", "",
         f"Judge `{s['judge_model']}` (effort {s['judge_effort']}), embeddings `{s['embeddings']}`, ragas 0.4.3. "
         "Built by `scripts/run_ragas_eval.py`; inputs and rules are in its docstring. Held-out set not touched.", "",
         "Faithfulness and answer relevancy: answered rows only (any scope). Context precision and recall: "
         "in-scope rows with a reference, any status; `not_covered` rows are scored on the parents retrieved "
         "but not used, `answered_partial` rows on the judge-filtered parents the generator saw.", "",
         "| Status | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for status in STATUS_ROWS + ("all",):
        L.append(f"| {status} | " + " | ".join(fmt(s["table"][status][m]) for m in METRICS) + " |")
    L += ["", "## 5 lowest-scoring ids per metric", ""]
    for m in METRICS:
        L.append(f"- **{m}**: " + ", ".join(f"{i} ({v:.2f})" for v, i in s["lowest"][m]))
    L += ["", "## Ragas faithfulness vs existing grades", "",
          "Mapping fixed before the run: 1.0 = SUPPORTED, 0.5 to <1.0 = PARTIALLY_SUPPORTED, <0.5 = UNSUPPORTED.", "",
          f"- Three-class agreement: **{pct(a['three_class'], a['n'])}**",
          f"- Binary (SUPPORTED vs not): **{pct(a['binary'], a['n'])}**",
          f"- Three-class by grader: manual {pct(*a['by_grader']['manual'])}, automated {pct(*a['by_grader']['automated'])}",
          "", "Confusion (existing -> ragas):", ""]
    L += [f"- {k}: {v}" for k, v in sorted(a["confusion"].items())]
    L += ["", "| id | status | ragas | ragas label | existing | grader |", "|---|---|---|---|---|---|"]
    L += [f"| {p['id']} | {p['status']} | {p['ragas']} | {p['ragas_label']} | {p['existing']} | {p['grader']} |"
          for p in a["pairs"]]
    if s["missing_scores"]:
        L += ["", f"**Missing scores** (failed, see scores.jsonl): {s['missing_scores']}"]
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="score only the first N scorable ids (smoke test)")
    p.add_argument("--report", action="store_true", help="rebuild the report from saved scores, no API calls")
    p.add_argument("--explain", type=int, nargs="+", metavar="ID",
                   help="print the statements ragas faithfulness marks unsupported for these ids (re-runs the judge)")
    args = p.parse_args()
    load_dotenv(".env")
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = build_dataset()
    if args.explain:
        asyncio.run(explain(rows, args.explain))
        return
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    usage = {} if args.report else asyncio.run(score_all(rows, args.limit))
    print("judge usage this run:", usage)
    write_report(summarize(rows, usage))


if __name__ == "__main__":
    main()
