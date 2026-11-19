"""Create deterministic synthetic transcripts, probes, summaries, and graph facts."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
personas = json.loads((ROOT / "personas.json").read_text())
sessions, probes, summaries, facts = [], [], [], []

for p in personas:
    pid, name = p["persona_id"], p["name"]
    timeline = p["timeline"]
    current = {k: next((x["value"] for x in reversed(v) if x["to_session"] is None), v[-1]["value"]) for k, v in timeline.items()}
    history = {k: v for k, v in timeline.items() if len(v) > 1}
    old_attr = next(iter(history), None)
    for sid in range(1, 4):
        text = f"{name} shared that they are a {p['description'].split(',')[0].lower()} and prefer {current.get('style', current.get('work_style', 'practical suggestions'))}."
        if sid == 1:
            text += " They asked for recommendations that respect their stated preferences."
        elif sid == 2:
            text += f" They gave an update: their {old_attr or 'plans'} are now {current.get(old_attr, 'unchanged') if old_attr else 'unchanged'}."
        else:
            text += f" They confirmed their {list(current)[0]} preference is {list(current.values())[0]}."
        sessions.append({"persona_id":pid,"session_id":sid,"transcript":[{"role":"user","text":text},{"role":"assistant","text":"I’ll keep that in mind for suggestions in this conversation."}]})
    for attr, versions in timeline.items():
        latest = versions[-1]["value"]
        stale = versions[0]["value"] if len(versions) > 1 else None
        facts.append({"persona_id":pid,"entity":name,"attribute":attr,"value":latest,"session_id":versions[-1]["from_session"],"superseded":False,"superseded_by":None,"aliases":[name.split()[0],"the user"]})
        for old in versions[:-1]:
            facts.append({"persona_id":pid,"entity":name,"attribute":attr,"value":old["value"],"session_id":old["from_session"],"superseded":True,"superseded_by":latest,"aliases":[name.split()[0],"the user"]})
    summary = f"{name}: {p['description']} Current remembered details: " + "; ".join(f"{k.replace('_',' ')}: {v}" for k,v in current.items()) + "."
    if old_attr:
        summary += f" Earlier {old_attr.replace('_',' ')} was {history[old_attr][0]['value']}; updated to {current[old_attr]} in session {history[old_attr][-1]['from_session']}."
    summaries.append({"persona_id":pid,"after_session":3,"summary_text":summary})
    # Four probe types exercise recall, update, history, and indirect personalization.
    recall_attr = next(iter(current))
    probes.extend([
        {"probe_id":f"{pid}-recall","persona_id":pid,"type":"recall","question":f"What should you remember about my {recall_attr.replace('_',' ')}?","ground_truth":str(current[recall_attr])},
        {"probe_id":f"{pid}-update","persona_id":pid,"type":"update","question":f"What is my current {old_attr.replace('_',' ') if old_attr else recall_attr.replace('_',' ')}?","ground_truth":str(current.get(old_attr, current[recall_attr])),"stale_answer":str(history[old_attr][0]['value']) if old_attr else None},
        {"probe_id":f"{pid}-history","persona_id":pid,"type":"history","question":f"Did my {old_attr.replace('_',' ') if old_attr else recall_attr.replace('_',' ')} ever change? What was it before?","ground_truth":f"{history[old_attr][0]['value']} before {current[old_attr]}" if old_attr else f"No change; {current[recall_attr]}"},
        {"probe_id":f"{pid}-indirect","persona_id":pid,"type":"indirect","question":"Suggest something I might enjoy, taking my preferences into account.","ground_truth":f"A suggestion suited to {', '.join(str(v) for v in current.values())}"}
    ])

for filename, rows in [("sessions.json",sessions),("probes.json",probes),("summaries.json",summaries),("graph_facts.json",facts)]:
    (ROOT / filename).write_text(json.dumps(rows, indent=2) + "\n")
