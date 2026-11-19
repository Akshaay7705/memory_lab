"""Seed MongoDB with synthetic data and lexical retrieval vectors; run manually, never from the app."""
import json
import sys
from pathlib import Path

import streamlit as st
from pymongo import MongoClient, UpdateOne

from memory_vectors import feature_vector

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def read_json(name):
    return json.loads((DATA / name).read_text())


def safe_error(exc, secrets_to_hide):
    detail = str(exc)
    for secret in secrets_to_hide:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    status = getattr(exc, "status_code", None)
    prefix = f"HTTP {status}: " if status else ""
    return f"{type(exc).__name__}: {prefix}{detail[:400]}"


def main():
    try:
        uri = st.secrets["MONGODB_URI"]
        database_name = st.secrets.get("MONGODB_DATABASE", "memory_lab")
    except Exception:
        print("Set MONGODB_URI in .streamlit/secrets.toml first.")
        return 1

    try:
        personas = read_json("personas.json")
        sessions = read_json("sessions.json")
        probes = read_json("probes.json")
        summaries = read_json("summaries.json")
        facts = read_json("graph_facts.json")
    except Exception:
        print("Synthetic seed files are missing or unreadable. Regenerate them with: python data/generate_data.py")
        return 1

    client = None
    stage = "MongoDB connection"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=7000)
        client.admin.command("ping")
        db = client[database_name]
        texts = []
        for session in sessions:
            text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in session["transcript"])
            texts.append(text)
        chunks = []
        for session, text in zip(sessions, texts):
            vector = feature_vector(text).tolist()
            chunks.append({"persona_id": session["persona_id"], "session_id": session["session_id"], "text": text, "embedding": vector})

        collections = {
            "personas": (personas, lambda d: d["persona_id"]),
            "sessions": (sessions, lambda d: f"{d['persona_id']}:{d['session_id']}"),
            "probes": (probes, lambda d: d["probe_id"]),
            "summaries": (summaries, lambda d: f"{d['persona_id']}:{d['after_session']}"),
            "graph_facts": (facts, lambda d: f"{d['persona_id']}:{d['attribute']}:{d['session_id']}:{d['value']}"),
            "chunks": (chunks, lambda d: f"{d['persona_id']}:{d['session_id']}"),
        }
        stage = "MongoDB writes"
        for name, (rows, key_fn) in collections.items():
            operations = [UpdateOne({"_id": key_fn(row)}, {"$set": row}, upsert=True) for row in rows]
            if operations:
                db[name].bulk_write(operations, ordered=False)
        db.chat_logs.create_index([("created_at", -1)])
        db.results.create_index([("strategy", 1), ("probe_type", 1)])
        print(f"Seeded {len(personas)} personas, {len(sessions)} sessions, {len(probes)} probes, {len(chunks)} retrieval vectors.")
        return 0
    except Exception as exc:
        print(f"Seeding failed during {stage}. {safe_error(exc, [uri])}")
        if stage == "MongoDB connection":
            if type(exc).__name__ == "InvalidURI":
                print("MongoDB rejected the URI format. Percent-encode the username and password with urllib.parse.quote_plus, then update MONGODB_URI. Do not encode the host or query string.")
            elif getattr(exc, "code", None) == 8000 or "authentication failed" in str(exc).lower():
                print("Atlas rejected authentication. In Atlas Database Access, reset or create a database user, then update MONGODB_URI with that exact username and password. URL-encode reserved characters in both credentials.")
            else:
                print("Check that Atlas is running, the URI is correct, and your current IP is allowed in Network Access.")
        else:
            print("Check the MongoDB user's write permissions and Atlas availability.")
        return 1
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
