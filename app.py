import re
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from groq import Groq
from pymongo import MongoClient

from config import MAX_LLM_CALLS_PER_SESSION
from memory_lab_core import probe_match, run_answer

st.set_page_config(page_title="Memory Lab", layout="wide")
STRATEGIES = {"No Memory": "no_memory", "Plain RAG": "rag", "Rolling Summary": "summary", "Graph Memory": "graph"}


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


def answer(strategy, persona_id, question):
    """Shared answer path: same provider, prompt, and bounded memory context for each strategy."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("Provider not configured")
        return run_answer(strategy, persona_id, question, get_db(), groq_client(key), st.session_state.get("_current_turns", []))
    except Exception as exc:
        raise RuntimeError("Answer request failed") from exc


def store_chat_log(persona_id, strategy, question, result):
    try:
        get_db().chat_logs.insert_one({"persona_id": persona_id, "strategy": strategy, "question": question,
                                      "answer": result["answer"], "latency_s": result["latency_s"],
                                      "created_at": datetime.now(timezone.utc)})
    except Exception:
        pass


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


BASELINE_STRATEGIES = ["No memory", "Plain RAG", "Rolling summary", "Graph memory"]
PROBE_TYPES = ["recall", "update", "history", "indirect"]


def normalize_strategy(value):
    name = re.sub(r"[_-]+", " ", str(value or "").strip().lower())
    compact = " ".join(name.split())
    if compact in {"no memory", "no memory baseline"}:
        return "No memory"
    if compact in {"rag", "plain rag"}:
        return "Plain RAG"
    if compact in {"summary", "rolling summary"}:
        return "Rolling summary"
    if compact in {"graph", "graph memory", "full graph", "full graph memory"}:
        return "Graph memory"
    return str(value or "Unknown strategy")


def normalize_probe_type(value):
    name = str(value or "").strip().lower()
    return name if name in PROBE_TYPES else name or "unknown"


def ablation_variant(value):
    name = re.sub(r"[_-]+", " ", str(value or "").strip().lower())
    if "supersession" in name and ("without" in name or "no supersession" in name or "ablation" in name):
        return "Without supersession"
    if "entity resolution" in name and ("without" in name or "no entity" in name or "ablation" in name):
        return "Without entity resolution"
    if name.strip() in {"graph", "graph memory", "full graph", "full graph memory"}:
        return "Full graph memory"
    return None


def correct_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and not np.isnan(value):
        return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "1", "correct"}:
            return True
        if value.strip().lower() in {"false", "no", "0", "incorrect"}:
            return False
    return None


def accuracy_summary(frame, strategy_column, strategy, probe_type=None):
    subset = frame[frame[strategy_column] == strategy]
    if probe_type is not None:
        subset = subset[subset["probe_type_normalized"] == probe_type]
    subset = subset[subset["correct_normalized"].notna()]
    count = len(subset)
    if count == 0:
        return None, 0, 0
    correct = int(subset["correct_normalized"].sum())
    return correct / count, correct, count


def render_accuracy_table(frame, strategy_column, strategies, probe_types, overall_label=True):
    import pandas as pd

    columns = list(probe_types) + (["overall"] if overall_label else [])
    display_rows, scores = [], []
    for strategy in strategies:
        values, numeric = {"Strategy": strategy}, {}
        for probe_type in columns:
            score, correct, count = accuracy_summary(frame, strategy_column, strategy, None if probe_type == "overall" else probe_type)
            title = probe_type.title() if probe_type != "overall" else "Overall"
            values[title] = f"{score:.0%} ({correct}/{count})" if score is not None else ""
            numeric[title] = score
        display_rows.append(values)
        scores.append(numeric)
    table = pd.DataFrame(display_rows)
    score_table = pd.DataFrame(scores)
    style = pd.DataFrame("", index=table.index, columns=table.columns)
    for column in score_table:
        available = score_table[column].dropna()
        if not available.empty:
            best = available.max()
            style.loc[score_table[column] == best, column] = "font-weight: bold"
    st.dataframe(table.style.apply(lambda _: style, axis=None), use_container_width=True, hide_index=True)
    return score_table


def render_insights():
    st.header("Insights")
    if st.button("Refresh results", key="refresh_insights"):
        load_insights.clear()
    try:
        rows = load_insights()
    except Exception:
        friendly_error("Results database")
        return
    if not rows:
        st.info("Results not loaded yet. After seeding, run `python evaluate.py` to evaluate the stored probes and populate this view.")
    else:
        import pandas as pd

        frame = pd.DataFrame(rows)
        if "strategy" in frame:
            frame["strategy_normalized"] = frame["strategy"].map(normalize_strategy)
            frame["variant"] = frame["strategy"].map(ablation_variant)
        else:
            frame["strategy_normalized"] = "Unknown strategy"
            frame["variant"] = None
        frame["probe_type_normalized"] = frame.get("probe_type", pd.Series(index=frame.index, dtype=object)).map(normalize_probe_type)
        frame["correct_normalized"] = frame.get("correct", pd.Series(index=frame.index, dtype=object)).map(correct_value)

        st.subheader("1. Accuracy by strategy and probe type")
        probe_counts = {}
        for probe_type in PROBE_TYPES:
            typed = frame[frame["probe_type_normalized"] == probe_type]
            probe_counts[probe_type] = int(typed["probe_id"].nunique()) if "probe_id" in typed else len(typed)
        st.caption("Unique probes represented in results: " + " · ".join(f"{kind.title()}: {count}" for kind, count in probe_counts.items()) + ". Table counts are result records, so reruns remain visible in the denominators.")
        render_accuracy_table(frame, "strategy_normalized", BASELINE_STRATEGIES, PROBE_TYPES)
        chart_rows = []
        for strategy in BASELINE_STRATEGIES:
            for probe_type in PROBE_TYPES:
                score, correct, count = accuracy_summary(frame, "strategy_normalized", strategy, probe_type)
                if score is not None:
                    chart_rows.append({"Strategy": strategy, "Probe type": probe_type.title(), "Accuracy": score})
        if chart_rows:
            st.bar_chart(pd.DataFrame(chart_rows), x="Strategy", y="Accuracy", color="Probe type")

        st.subheader("2. Cost and latency")
        metric_columns = ["latency_s", "tokens_in", "tokens_out", "cost_usd"]
        for column in metric_columns:
            if column not in frame:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        metric_rows = []
        for strategy in BASELINE_STRATEGIES:
            subset = frame[frame["strategy_normalized"] == strategy]
            metric_rows.append({
                "Strategy": strategy,
                "Avg latency (s)": subset["latency_s"].mean(),
                "Avg tokens in": subset["tokens_in"].mean(),
                "Avg tokens out": subset["tokens_out"].mean(),
                "Avg cost per query ($)": subset["cost_usd"].mean(),
            })
        metric_table = pd.DataFrame(metric_rows)
        st.dataframe(metric_table.style.format({"Avg latency (s)": "{:.2f}", "Avg tokens in": "{:.0f}", "Avg tokens out": "{:.0f}", "Avg cost per query ($)": "${:.5f}"}, na_rep=""), use_container_width=True, hide_index=True)

        setup_fields = ["setup_cost_usd", "setup_tokens_in", "setup_tokens_out"]
        available_setup_fields = [column for column in setup_fields if column in frame and frame[column].notna().any()]
        if available_setup_fields:
            setup = frame[frame[available_setup_fields].notna().any(axis=1)].copy()
            for column in available_setup_fields:
                setup[column] = pd.to_numeric(setup[column], errors="coerce")
            setup["run_key"] = setup["run_id"] if "run_id" in setup else setup.index.astype(str)
            setup = setup.drop_duplicates(["strategy_normalized", "run_key"])
            notes = []
            for strategy, group in setup.groupby("strategy_normalized"):
                parts = []
                if "setup_cost_usd" in available_setup_fields:
                    parts.append(f"${group['setup_cost_usd'].sum():.5f} setup cost")
                if "setup_tokens_in" in available_setup_fields:
                    parts.append(f"{group['setup_tokens_in'].sum():.0f} setup input tokens")
                if "setup_tokens_out" in available_setup_fields:
                    parts.append(f"{group['setup_tokens_out'].sum():.0f} setup output tokens")
                notes.append(f"{strategy}: {', '.join(parts)}")
            st.caption("One-time setup, reported separately and excluded from per-query averages: " + "; ".join(notes) + ".")
        else:
            st.caption("One-time setup costs/tokens are not recorded in results; per-query averages exclude them.")

        st.subheader("3. Ablations (graph memory only)")
        ablation_order = ["Full graph memory", "Without supersession", "Without entity resolution"]
        ablation_frame = frame[frame["variant"].isin(ablation_order)]
        present_variants = [variant for variant in ablation_order if variant in set(ablation_frame["variant"])]
        if present_variants:
            render_accuracy_table(ablation_frame, "variant", present_variants, ["update", "indirect"], overall_label=True)
            findings = []
            comparisons = [
                ("Without supersession", "update", "update questions"),
                ("Without entity resolution", "indirect", "indirect-reference questions"),
            ]
            for variant, probe_type, target in comparisons:
                full_score, _, _ = accuracy_summary(ablation_frame, "variant", "Full graph memory", probe_type)
                ablated_score, _, _ = accuracy_summary(ablation_frame, "variant", variant, probe_type)
                if full_score is not None and ablated_score is not None and ablated_score >= full_score:
                    findings.append(f"{variant} did not lower accuracy on {target} ({ablated_score:.0%} vs {full_score:.0%} for full graph memory).")
            if findings:
                st.caption("Ablation findings: " + " ".join(findings))
        else:
            st.caption("No graph ablation results are present.")

    #     st.subheader("Failure cases")
    #     failures = frame[frame["correct_normalized"] == False] if "correct_normalized" in frame else frame.iloc[0:0]
    #     for row in failures.head(25).to_dict("records"):
    #         st.markdown(f"**{row.get('strategy','')} · {row.get('probe_id','')}** — {row.get('question','')}")
    #         st.write(f"Ground truth: {row.get('ground_truth','')} · Answer: {row.get('answer','')} · Cause: {row.get('cause','Not specified')}")
    
    
    
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


st.title("Memory Lab")
st.write("Does graph memory beat plain RAG for long-term personalization, and at what cost?")
st.caption("First load may take a minute. All personas are synthetic.")
st.markdown(
    '### Want to contribute or check out real user data? '
    '[Click here](https://memory-lab-user.streamlit.app/)'
)
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
