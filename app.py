import re
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from groq import Groq
from pymongo import MongoClient

from config import GROQ_MODEL, CONTEXT_TOKEN_BUDGET, RAG_TOP_K, MAX_LLM_CALLS_PER_SESSION
from memory_vectors import feature_vector

st.set_page_config(page_title="Memory Lab", page_icon="🧠", layout="wide")
STRATEGIES = {"No Memory": "no_memory", "Plain RAG": "rag", "Rolling Summary": "summary", "Graph Memory": "graph"}
SYSTEM_PROMPT = "You are a helpful chat assistant in a long-term memory research demo. Use only context provided below as remembered facts. If the context does not answer the question, say so plainly. Treat current facts as authoritative and use superseded facts only when asked about history. Keep responses clear and avoid claiming certainty beyond the evidence.\n\nMEMORY CONTEXT:\n{context}"


def friendly_error(label):
    st.warning(f"{label} is temporarily unavailable. Please try again shortly.")


@st.cache_resource(show_spinner=False)
def mongo_client(uri):
    return MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)


@st.cache_resource(show_spinner=False)
def groq_client(api_key):
    return Groq(api_key=api_key, timeout=30, max_retries=1)


def get_db():
    uri = st.secrets.get("MONGODB_URI", "")
    if not uri:
        raise RuntimeError("MongoDB is not configured")
    return mongo_client(uri)[st.secrets.get("MONGODB_DATABASE", "memory_lab")]


@st.cache_data(ttl=60, show_spinner=False)
def load_personas():
    return list(get_db().personas.find({}, {"_id": 0}))


@st.cache_data(ttl=60, show_spinner=False)
def load_probes(persona_id):
    return list(get_db().probes.find({"persona_id": persona_id}, {"_id": 0}).sort("type", 1))


@st.cache_data(ttl=60, show_spinner=False)
def load_insights():
    return list(get_db().results.find({}, {"_id": 0}))


def clip_context(text):
    # A conservative word-based approximation keeps each strategy within the shared context allowance.
    return " ".join(text.split()[: CONTEXT_TOKEN_BUDGET * 3 // 4])


def build_memory(strategy, persona_id, question):
    try:
        return _build_memory(strategy, persona_id, question)
    except Exception as exc:
        raise RuntimeError("Memory lookup failed") from exc


def _build_memory(strategy, persona_id, question):
    if strategy == "no_memory":
        return "No long-term memory is available for this strategy.", {"kind": "none", "items": []}
    db = get_db()
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
        items = [{"session_id": d.get("session_id"), "similarity": round(score, 3), "text": d.get("text", "")} for score, d in selected]
        return clip_context("\n".join(x["text"] for x in items) or "No matching history was found."), {"kind": "Retrieved chunks", "items": items}
    if strategy == "summary":
        doc = db.summaries.find_one({"persona_id": persona_id}, {"_id": 0}, sort=[("after_session", -1)])
        summary = (doc or {}).get("summary_text", "No stored summary was found.")
        return clip_context(summary), {"kind": "Stored rolling summary", "items": [{"text": summary}]}
    if strategy == "graph":
        all_facts = list(db.graph_facts.find({"persona_id": persona_id}, {"_id": 0}))
        question_lower = question.lower()
        terms = set(re.findall(r"[a-z0-9]+", question_lower))
        hits = [f for f in all_facts if any(alias.lower() in terms for alias in f.get("aliases", [])) or f.get("attribute", "").replace("_", " ").lower() in question_lower]
        chosen = hits or all_facts
        items = [{"entity": f.get("entity"), "attribute": f.get("attribute"), "value": f.get("value"), "status": "superseded" if f.get("superseded") else "current", "superseded_by": f.get("superseded_by"), "session_id": f.get("session_id")} for f in chosen]
        context = "\n".join(f"{x['entity']} | {x['attribute']}: {x['value']} ({x['status']}{'; updated to ' + str(x['superseded_by']) if x.get('superseded_by') else ''})" for x in items)
        return clip_context(context or "No relevant facts were found."), {"kind": "Graph facts", "items": items}
    raise ValueError("Unknown strategy")


def answer(strategy, persona_id, question):
    """Shared answer path: same provider, prompt, and bounded memory context for each strategy."""
    started = time.perf_counter()
    context, inspector = build_memory(strategy, persona_id, question)
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("Provider not configured")
        response = groq_client(key).chat.completions.create(
            model=GROQ_MODEL,
            messages=([{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
                     + st.session_state.get("_current_turns", [])
                     + [{"role": "user", "content": question}]),
            temperature=0.2,
        )
        elapsed = time.perf_counter() - started
        usage = response.usage
        inspector["context_text"] = context
        inspector["current_chat_turns"] = st.session_state.get("_current_turns", [])
        return {"answer": response.choices[0].message.content or "I could not form an answer.", "latency_s": elapsed,
                "tokens_in": usage.prompt_tokens if usage else 0, "tokens_out": usage.completion_tokens if usage else 0,
                "inspector": inspector}
    except Exception as exc:
        raise RuntimeError("Answer request failed") from exc


def store_chat_log(persona_id, strategy, question, result):
    try:
        get_db().chat_logs.insert_one({"persona_id": persona_id, "strategy": strategy, "question": question,
                                      "answer": result["answer"], "latency_s": result["latency_s"],
                                      "created_at": datetime.now(timezone.utc)})
    except Exception:
        pass


def probe_match(answer_text, ground_truth):
    truth = set(re.findall(r"[a-z0-9]+", (ground_truth or "").lower()))
    if not truth:
        return False
    # This is a visible lexical heuristic, not a semantic evaluation judge.
    answer_words = set(re.findall(r"[a-z0-9]+", (answer_text or "").lower()))
    return len(truth & answer_words) / len(truth) >= 0.5


def render_chat(tab_key, strategy, persona_id, probes):
    messages = st.session_state.setdefault(f"messages_{tab_key}", [])
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe.get("type", "other"), []).append(probe)
    options = [f"[{kind}] {probe['question']}" for kind, group in grouped.items() for probe in group]
    chosen = st.selectbox("Suggested probe questions", ["Choose a probe…"] + options, key=f"probe_{tab_key}")
    selected = next((p for group in grouped.values() for p in group if f"[{p.get('type')}] {p['question']}" == chosen), None)
    question = st.text_area("Message", key=f"input_{tab_key}", placeholder="Ask a question…", height=85, max_chars=2000)
    c1, c2 = st.columns(2)
    send = c1.button("Send message", key=f"send_{tab_key}", type="primary")
    use_probe = c2.button("Ask selected probe", key=f"ask_probe_{tab_key}", disabled=selected is None)
    prompt = selected["question"] if use_probe and selected else question
    current_probe = selected if use_probe else None
    if (send and question.strip()) or (use_probe and selected):
        if st.session_state.get("llm_calls", 0) >= MAX_LLM_CALLS_PER_SESSION:
            st.info(f"This session has reached the {MAX_LLM_CALLS_PER_SESSION}-answer demo limit. Start a fresh session to continue.")
        else:
            try:
                st.session_state["llm_calls"] = st.session_state.get("llm_calls", 0) + 1
                st.session_state["_current_turns"] = [{"role": m["role"], "content": m["content"]} for m in messages[-8:]]
                result = answer(strategy, persona_id, prompt.strip())
                messages.extend([{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": result["answer"], "meta": result, "probe": current_probe}])
                # Only prewritten synthetic probes are logged; free-form visitor text may contain personal data.
                if current_probe:
                    store_chat_log(persona_id, strategy, prompt.strip(), result)
            except Exception:
                friendly_error("The assistant")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Conversation")
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    st.caption(f"Latency {meta['latency_s']:.2f}s · {meta['tokens_in']} input / {meta['tokens_out']} output tokens")
                    probe = msg.get("probe")
                    if probe:
                        matched = probe_match(msg["content"], probe.get("ground_truth", ""))
                        st.info(f"Ground truth: {probe.get('ground_truth')}\n\nLexical match heuristic: {'matched' if matched else 'not matched'}")
    with right:
        st.subheader("Memory inspector")
        latest = next((m.get("meta") for m in reversed(messages) if m.get("role") == "assistant" and m.get("meta")), None)
        if latest:
            st.caption(latest["inspector"]["kind"])
            st.markdown("**Exact long-term context**")
            st.code(latest["inspector"].get("context_text", ""), language=None)
            if latest["inspector"].get("current_chat_turns"):
                with st.expander("Current chat turns sent to the model"):
                    for turn in latest["inspector"]["current_chat_turns"]:
                        st.write(f"**{turn['role']}**: {turn['content']}")
            for item in latest["inspector"]["items"]:
                st.json(item, expanded=True)
        else:
            st.caption("The exact memory context used for the latest answer will appear here.")


def render_insights():
    st.header("Insights")
    try:
        rows = load_insights()
    except Exception:
        friendly_error("Results database")
        return
    if not rows:
        st.info("Results not loaded yet.")
    else:
        import pandas as pd
        frame = pd.DataFrame(rows)
        if {"strategy", "probe_type", "correct"}.issubset(frame.columns):
            grouped = frame.groupby(["strategy", "probe_type"], dropna=False).agg(accuracy=("correct", "mean"), queries=("correct", "size")).reset_index()
            st.subheader("Accuracy by strategy and probe type")
            st.dataframe(grouped, use_container_width=True, hide_index=True)
            st.bar_chart(grouped, x="strategy", y="accuracy", color="probe_type")
        if {"strategy", "latency_s", "cost_usd"}.issubset(frame.columns):
            st.subheader("Latency and cost per query")
            st.dataframe(frame.groupby("strategy").agg(latency_s=("latency_s", "mean"), cost_usd=("cost_usd", "mean")).reset_index(), use_container_width=True, hide_index=True)
        ablations = frame[frame["strategy"].astype(str).str.contains("ablation|without", case=False)] if "strategy" in frame else frame.iloc[0:0]
        st.subheader("Ablations")
        if not ablations.empty:
            st.dataframe(ablations, use_container_width=True, hide_index=True)
        else:
            st.caption("No ablation results are present.")
        st.subheader("Failure cases")
        failures = frame[frame["correct"] == False] if "correct" in frame else frame.iloc[0:0]
        for row in failures.head(25).to_dict("records"):
            st.markdown(f"**{row.get('strategy','')} · {row.get('probe_id','')}** — {row.get('question','')}")
            st.write(f"Ground truth: {row.get('ground_truth','')} · Answer: {row.get('answer','')} · Cause: {row.get('cause','Not specified')}")
    st.subheader("Limitations")
    st.write("Synthetic personas; LLM-written transcripts; small sample (10 personas, N probes); single LLM; fixed attributes; lexical hash vectors for Plain RAG; no real users yet.")
    try:
        db = get_db()
        feedback_exists = "pilot_feedback" in db.list_collection_names() and db.pilot_feedback.estimated_document_count() > 0
        if feedback_exists:
            st.subheader("Pilot feedback")
            st.write("Pilot feedback is available in the pilot_feedback collection.")
    except Exception:
        pass
    timestamps = [row.get("updated_at") or row.get("created_at") for row in rows if isinstance(row.get("updated_at") or row.get("created_at"), datetime)] if rows else []
    latest = max(timestamps) if timestamps else None
    if latest and latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    st.caption(f"Last updated: {latest.astimezone().strftime('%Y-%m-%d %H:%M %Z') if latest else 'unknown'}")


st.title("🧠 Memory Lab")
st.write("Does graph memory beat plain RAG for long-term personalization, and at what cost?")
st.caption("First load may take a minute. All personas are synthetic.")
try:
    personas = load_personas()
except Exception:
    personas = []
    friendly_error("Persona database")
if not personas:
    st.info("Connect a seeded MongoDB database to begin. See the README for setup.")
    st.stop()
labels = {p["persona_id"]: f"{p['name']} — {p.get('description', '')}" for p in personas}
ids = list(labels)
if st.session_state.get("persona_id") not in ids:
    st.session_state["persona_id"] = ids[0]
selection = st.selectbox("Persona", ids, format_func=lambda pid: labels.get(pid, pid), key="persona_id")
previous_persona = st.session_state.get("active_persona_id")
if previous_persona and previous_persona != selection:
    for strategy_name in STRATEGIES.values():
        st.session_state.pop(f"messages_{strategy_name}", None)
st.session_state["active_persona_id"] = selection
try:
    probes = load_probes(selection)
except Exception:
    probes = []
    friendly_error("Probe database")
tabs = st.tabs(["No Memory", "Plain RAG", "Rolling Summary", "Graph Memory", "Insights"])
for tab, label in zip(tabs[:4], list(STRATEGIES)):
    with tab:
        st.header(label)
        try:
            render_chat(STRATEGIES[label], STRATEGIES[label], selection, probes)
        except Exception:
            friendly_error("This strategy")
with tabs[4]:
    try:
        render_insights()
    except Exception:
        friendly_error("Insights")
