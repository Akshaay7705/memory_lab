"""Shared retrieval and answer logic for the app and offline evaluator."""
import time
import re

import numpy as np

from config import GROQ_MODEL, CONTEXT_TOKEN_BUDGET, RAG_TOP_K, MAX_OUTPUT_TOKENS
from memory_vectors import feature_vector

SYSTEM_PROMPT = "You are a helpful chat assistant in a long-term memory research demo. Use only context provided below as remembered facts. If the context does not answer the question, say so plainly. Treat current facts as authoritative and use superseded facts only when asked about history. Keep responses clear and avoid claiming certainty beyond the evidence.\n\nMEMORY CONTEXT:\n{context}"


def clip_context(text):
    return " ".join(text.split()[: CONTEXT_TOKEN_BUDGET * 3 // 4])


def build_memory(db, strategy, persona_id, question):
    if strategy == "no_memory":
        return "No long-term memory is available for this strategy.", {"kind": "none", "items": []}
    if strategy == "rag":
        qvec = feature_vector(question).astype(float)
        docs = list(db.chunks.find({"persona_id": persona_id}, {"_id": 0}).limit(500))
        scored = []
        qnorm = np.linalg.norm(qvec)
        for doc in docs:
            vec = np.asarray(doc.get("embedding", []), dtype=float)
            denom = qnorm * np.linalg.norm(vec)
            similarity = float(np.dot(qvec, vec) / denom) if denom else 0.0
            scored.append((similarity, doc))
        selected = sorted(scored, key=lambda x: x[0], reverse=True)[:RAG_TOP_K]
        items = [{"session_id": doc.get("session_id"), "similarity": round(score, 3), "text": doc.get("text", "")} for score, doc in selected]
        context = "\n".join(item["text"] for item in items) or "No matching history was found."
        return clip_context(context), {"kind": "Retrieved chunks", "items": items}
    if strategy == "summary":
        doc = db.summaries.find_one({"persona_id": persona_id}, {"_id": 0}, sort=[("after_session", -1)])
        summary = (doc or {}).get("summary_text", "No stored summary was found.")
        return clip_context(summary), {"kind": "Stored rolling summary", "items": [{"text": summary}]}
    if strategy == "graph":
        facts = list(db.graph_facts.find({"persona_id": persona_id}, {"_id": 0}))
        lower_question = question.lower()
        terms = set(lower_question.replace("?", "").split())
        hits = [fact for fact in facts if any(alias.lower() in terms for alias in fact.get("aliases", [])) or fact.get("attribute", "").replace("_", " ").lower() in lower_question]
        selected = hits or facts
        items = [{"entity": fact.get("entity"), "attribute": fact.get("attribute"), "value": fact.get("value"),
                  "status": "superseded" if fact.get("superseded") else "current", "superseded_by": fact.get("superseded_by"),
                  "session_id": fact.get("session_id")} for fact in selected]
        context = "\n".join(f"{item['entity']} | {item['attribute']}: {item['value']} ({item['status']}{'; updated to ' + str(item['superseded_by']) if item.get('superseded_by') else ''})" for item in items)
        return clip_context(context or "No relevant facts were found."), {"kind": "Graph facts", "items": items}
    raise ValueError("Unknown strategy")


def run_answer(strategy, persona_id, question, db, client, current_turns=None):
    started = time.perf_counter()
    context, inspector = build_memory(db, strategy, persona_id, question)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=([{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
                      + (current_turns or []) + [{"role": "user", "content": question}]),
            temperature=0.2,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        usage = response.usage
        inspector["context_text"] = context
        inspector["current_chat_turns"] = current_turns or []
        return {"answer": response.choices[0].message.content or "I could not form an answer.",
                "latency_s": time.perf_counter() - started,
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
                "inspector": inspector}
    except Exception as exc:
        raise RuntimeError("Answer request failed") from exc


def probe_match(answer_text, ground_truth):
    truth = set(re.findall(r"[a-z0-9]+", (ground_truth or "").lower()))
    if not truth:
        return False
    answer_words = set(re.findall(r"[a-z0-9]+", (answer_text or "").lower()))
    return len(truth & answer_words) / len(truth) >= 0.5
