"""Run the synthetic probe suite and write measured results to MongoDB."""
import argparse
import sys
import time
import uuid
from datetime import datetime, timezone

import streamlit as st
from groq import Groq
from pymongo import MongoClient

from config import GROQ_INPUT_COST_PER_MILLION, GROQ_OUTPUT_COST_PER_MILLION
from memory_lab_core import probe_match, run_answer

STRATEGIES = ["no_memory", "rag", "summary", "graph"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N probes (useful for a small initial run).")
    parser.add_argument("--delay", type=float, default=2.1, help="Seconds between Groq calls; default stays near 30 calls per minute.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1.")
        return 1

    try:
        uri = st.secrets["MONGODB_URI"]
        api_key = st.secrets["GROQ_API_KEY"]
        database_name = st.secrets.get("MONGODB_DATABASE", "memory_lab")
    except Exception:
        print("Set MONGODB_URI and GROQ_API_KEY in .streamlit/secrets.toml first.")
        return 1

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=7000)
        client.admin.command("ping")
        db = client[database_name]
        probes = list(db.probes.find({}, {"_id": 0}).sort([("persona_id", 1), ("type", 1)]))
        if args.limit:
            probes = probes[:args.limit]
        if not probes:
            print("No probes found. Run python seed_db.py first.")
            return 1
        groq = Groq(api_key=api_key, timeout=45, max_retries=1)
        run_id = str(uuid.uuid4())
        result_rows = []
        total = len(probes) * len(STRATEGIES)
        step = 0
        for probe in probes:
            for strategy in STRATEGIES:
                if step:
                    time.sleep(max(args.delay, 0))
                step += 1
                result = run_answer(strategy, probe["persona_id"], probe["question"], db, groq)
                matched = probe_match(result["answer"], probe.get("ground_truth", ""))
                cost = (result["tokens_in"] * GROQ_INPUT_COST_PER_MILLION + result["tokens_out"] * GROQ_OUTPUT_COST_PER_MILLION) / 1_000_000
                result_rows.append({
                    "run_id": run_id,
                    "strategy": strategy,
                    "persona_id": probe["persona_id"],
                    "probe_id": probe["probe_id"],
                    "probe_type": probe["type"],
                    "question": probe["question"],
                    "ground_truth": probe.get("ground_truth", ""),
                    "answer": result["answer"],
                    "correct": matched,
                    "latency_s": result["latency_s"],
                    "tokens_in": result["tokens_in"],
                    "tokens_out": result["tokens_out"],
                    "cost_usd": cost,
                    "cause": "Lexical match heuristic did not find enough ground-truth terms." if not matched else None,
                    "created_at": datetime.now(timezone.utc),
                })
                print(f"Evaluated {step}/{total} probes and strategies.", flush=True)
        db.results.insert_many(result_rows, ordered=True)
        print(f"Stored {len(result_rows)} measured results in the results collection (run {run_id}).")
        print("Correctness uses the same lexical match heuristic shown in the chat UI; it is not a semantic judge.")
        return 0
    except Exception as exc:
        print(f"Evaluation failed ({type(exc).__name__}). No result batch was written. Check MongoDB/Groq access and retry.")
        return 1
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
