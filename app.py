"""
GenAI Impact Lab - Flask app serving logistic.pkl
Run locally :  python app.py
Render      :  Start command ->  gunicorn app:app
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]

# ---------------------------------------------------------------------------
# IMPORTANT: the model is a plain LogisticRegression with 11 NUMERIC inputs.
# Text columns were therefore converted to numbers before training. The codes
# below assume alphabetical LabelEncoder order (0, 1, 2 ...). If you encoded
# your training data differently, edit the lists/ranges here so they match.
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Major_Category": ["Arts", "Business", "Engineering", "Science", "Social Science"],
    "Primary_Use_Case": ["Brainstorming", "Coding", "Research", "Summarizing", "Writing"],
    "Paid_Subscription": ["No", "Yes"],
    "Institutional_Policy": ["Allowed", "Banned", "Restricted"],
}

# name -> (label, type, min, max, step, default, icon, hint)
FIELDS = [
    {"name": "Major_Category", "label": "Major", "type": "select", "icon": "🎓",
     "options": CATEGORIES["Major_Category"], "default": 0},
    {"name": "Year_of_Study", "label": "Year of study", "type": "range", "icon": "📅",
     "min": 1, "max": 6, "step": 1, "default": 2},
    {"name": "Pre_Semester_GPA", "label": "Pre-semester GPA", "type": "range", "icon": "📈",
     "min": 0, "max": 4, "step": 0.05, "default": 3.0},
    {"name": "Weekly_GenAI_Hours", "label": "Weekly GenAI hours", "type": "range", "icon": "🤖",
     "min": 0, "max": 60, "step": 0.5, "default": 10},
    {"name": "Primary_Use_Case", "label": "Primary use case", "type": "select", "icon": "🧩",
     "options": CATEGORIES["Primary_Use_Case"], "default": 0},
    {"name": "Prompt_Engineering_Skill", "label": "Prompt-engineering skill", "type": "range", "icon": "⚡",
     "min": 1, "max": 10, "step": 1, "default": 5},
    {"name": "Paid_Subscription", "label": "Paid subscription", "type": "select", "icon": "💳",
     "options": CATEGORIES["Paid_Subscription"], "default": 0},
    {"name": "Traditional_Study_Hours", "label": "Traditional study hours / week", "type": "range", "icon": "📚",
     "min": 0, "max": 60, "step": 0.5, "default": 12},
    {"name": "Perceived_AI_Dependency", "label": "Perceived AI dependency", "type": "range", "icon": "🔗",
     "min": 1, "max": 10, "step": 1, "default": 5},
    {"name": "Institutional_Policy", "label": "Institution policy on AI", "type": "select", "icon": "🏛️",
     "options": CATEGORIES["Institutional_Policy"], "default": 0},
    {"name": "Skill_Retention_Score", "label": "Skill retention score", "type": "range", "icon": "🧠",
     "min": 0, "max": 100, "step": 1, "default": 60},
]
FIELD_BY_NAME = {f["name"]: f for f in FIELDS}

app = Flask(__name__)


def to_number(field, raw):
    """Convert incoming form value to the numeric code the model expects."""
    if field["type"] == "select":
        # select values arrive as the option index (the label encoder code)
        return float(raw)
    return float(raw)


def run_prediction(payload):
    row = {}
    for name in FEATURES:
        if name not in payload:
            raise ValueError(f"Missing field: {name}")
        row[name] = to_number(FIELD_BY_NAME[name], payload[name])

    X = pd.DataFrame([row], columns=FEATURES)
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))

    # Per-feature pull toward the predicted class (coef * value) - simple explainability
    contrib = model.coef_[idx] * X.iloc[0].values
    drivers = sorted(
        [{"feature": n.replace("_", " "), "impact": float(c)} for n, c in zip(FEATURES, contrib)],
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )[:5]

    return {
        "prediction": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "drivers": drivers,
    }


@app.route("/")
def home():
    return render_template_string(
        PAGE.replace("__FIELDS__", json.dumps(FIELDS)).replace("__CLASSES__", json.dumps(CLASSES))
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        return jsonify(run_prediction(payload))
    except Exception as exc:  # return a readable error to the UI
        return jsonify({"error": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURES, "classes": CLASSES})


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenAI Impact Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#07070f; --card:rgba(255,255,255,.055); --line:rgba(255,255,255,.12);
  --text:#ecebff; --muted:#9a98c4; --a:#7c5cff; --b:#00e0c6; --c:#ff5c93;
  --High:#00e0c6; --Medium:#ffc857; --Low:#ff5c93;
}
*{box-sizing:border-box;margin:0}
body{font-family:'Space Grotesk',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.blob{position:fixed;border-radius:50%;filter:blur(90px);opacity:.45;z-index:0;animation:float 16s ease-in-out infinite}
.b1{width:420px;height:420px;background:var(--a);top:-120px;left:-100px}
.b2{width:380px;height:380px;background:var(--b);bottom:-120px;right:-80px;animation-delay:-6s}
.b3{width:300px;height:300px;background:var(--c);top:45%;left:55%;animation-delay:-11s;opacity:.25}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,-50px) scale(1.15)}}
.wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 20px 60px}
header{text-align:center;margin-bottom:34px}
.tag{display:inline-block;font:600 12px 'JetBrains Mono',monospace;letter-spacing:.14em;color:var(--b);border:1px solid var(--line);padding:6px 14px;border-radius:99px;background:var(--card)}
h1{font-size:clamp(34px,6vw,60px);line-height:1.05;margin:16px 0 10px;font-weight:700;
   background:linear-gradient(100deg,#fff 10%,var(--a) 45%,var(--b) 90%);-webkit-background-clip:text;background-clip:text;color:transparent}
header p{color:var(--muted);max-width:560px;margin:auto}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:22px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:24px;backdrop-filter:blur(18px)}
.card h2{font-size:15px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:18px;font-weight:500}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:16px 18px}
@media(max-width:560px){.fields{grid-template-columns:1fr}}
.f label{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-bottom:8px}
.f label b{color:var(--text);font:600 13px 'JetBrains Mono',monospace}
select{width:100%;padding:11px 12px;border-radius:12px;background:rgba(0,0,0,.35);border:1px solid var(--line);color:var(--text);font:inherit}
input[type=range]{-webkit-appearance:none;width:100%;height:6px;border-radius:9px;background:rgba(255,255,255,.14);outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,var(--a),var(--b));cursor:pointer;box-shadow:0 0 14px var(--a)}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border:0;border-radius:50%;background:var(--a);cursor:pointer}
button{margin-top:24px;width:100%;padding:16px;border:0;border-radius:14px;font:700 16px 'Space Grotesk';color:#07070f;cursor:pointer;
  background:linear-gradient(100deg,var(--a),var(--b));transition:transform .15s,box-shadow .2s;box-shadow:0 8px 30px -8px var(--a)}
button:hover{transform:translateY(-2px);box-shadow:0 14px 36px -8px var(--b)}
button:disabled{opacity:.6;cursor:wait}
.result{text-align:center}
.empty{color:var(--muted);padding:50px 10px;font-size:15px}
.empty .big{font-size:48px;display:block;margin-bottom:10px}
.ring{--p:0;--col:var(--a);width:190px;height:190px;border-radius:50%;margin:6px auto 14px;display:grid;place-items:center;
  background:conic-gradient(var(--col) calc(var(--p)*1%),rgba(255,255,255,.1) 0);transition:.8s}
.ring div{width:158px;height:158px;border-radius:50%;background:#0d0d1c;display:grid;place-items:center;align-content:center}
.ring .lbl{font-size:34px;font-weight:700}
.ring .pct{font:600 13px 'JetBrains Mono',monospace;color:var(--muted)}
.pbar{text-align:left;margin:12px 0}
.pbar .row{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px;color:var(--muted)}
.pbar .track{height:9px;background:rgba(255,255,255,.1);border-radius:9px;overflow:hidden}
.pbar .fill{height:100%;width:0;border-radius:9px;transition:width .9s cubic-bezier(.2,.8,.2,1)}
.drivers{text-align:left;margin-top:22px;border-top:1px solid var(--line);padding-top:16px}
.drivers h3{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-weight:500}
.drv{display:flex;justify-content:space-between;font-size:13px;padding:6px 0}
.drv span:last-child{font-family:'JetBrains Mono',monospace}
.pos{color:var(--b)}.neg{color:var(--c)}
.err{color:var(--c);padding:20px}
footer{text-align:center;color:var(--muted);font-size:12px;margin-top:30px}
.pop{animation:pop .5s cubic-bezier(.2,1.4,.4,1)}
@keyframes pop{from{transform:scale(.85);opacity:0}to{transform:scale(1);opacity:1}}
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
<div class="wrap">
  <header>
    <span class="tag">● LOGISTIC REGRESSION · LIVE MODEL</span>
    <h1>GenAI Impact Lab</h1>
    <p>Tune the sliders, describe a student profile, and watch the model predict the outcome level with full class probabilities.</p>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Student profile</h2>
      <div class="fields" id="fields"></div>
      <button id="go">Run prediction ✦</button>
    </div>

    <div class="card result" id="result">
      <h2>Model output</h2>
      <div class="empty"><span class="big">🔮</span>Your prediction will appear here.</div>
    </div>
  </div>
  <footer>Built with Flask · scikit-learn · deployed on Render</footer>
</div>

<script>
const FIELDS = __FIELDS__;
const CLASSES = __CLASSES__;
const COLORS = {High:'var(--High)',Medium:'var(--Medium)',Low:'var(--Low)'};
const $ = s => document.querySelector(s);

// build the form
FIELDS.forEach(f=>{
  const d = document.createElement('div'); d.className='f';
  if(f.type==='select'){
    d.innerHTML = `<label><span>${f.icon} ${f.label}</span></label>
      <select id="${f.name}">${f.options.map((o,i)=>`<option value="${i}" ${i===f.default?'selected':''}>${o}</option>`).join('')}</select>`;
  } else {
    d.innerHTML = `<label><span>${f.icon} ${f.label}</span><b id="v_${f.name}">${f.default}</b></label>
      <input type="range" id="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">`;
  }
  $('#fields').appendChild(d);
  if(f.type==='range') d.querySelector('input').addEventListener('input',e=>$('#v_'+f.name).textContent=e.target.value);
});

$('#go').addEventListener('click', async ()=>{
  const btn=$('#go'); btn.disabled=true; btn.textContent='Thinking…';
  const body={}; FIELDS.forEach(f=>body[f.name]=$('#'+f.name).value);
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json();
    if(!r.ok) throw new Error(j.error||'Prediction failed');
    render(j);
  }catch(e){
    $('#result').innerHTML = `<h2>Model output</h2><div class="err">⚠ ${e.message}</div>`;
  }
  btn.disabled=false; btn.textContent='Run prediction ✦';
});

function render(j){
  const col = COLORS[j.prediction] || 'var(--a)';
  const pct = (j.confidence*100).toFixed(1);
  const bars = CLASSES.map(c=>{
    const p=(j.probabilities[c]*100).toFixed(1);
    return `<div class="pbar"><div class="row"><span>${c}</span><span>${p}%</span></div>
      <div class="track"><div class="fill" data-w="${p}" style="background:${COLORS[c]||'var(--a)'}"></div></div></div>`;
  }).join('');
  const drv = j.drivers.map(d=>`<div class="drv"><span>${d.feature}</span>
      <span class="${d.impact>=0?'pos':'neg'}">${d.impact>=0?'+':''}${d.impact.toFixed(2)}</span></div>`).join('');
  $('#result').innerHTML = `<h2>Model output</h2>
    <div class="ring pop" style="--col:${col}" id="ring"><div><span class="lbl" style="color:${col}">${j.prediction}</span><span class="pct">${pct}% confident</span></div></div>
    ${bars}
    <div class="drivers"><h3>Top influencing factors</h3>${drv}</div>`;
  requestAnimationFrame(()=>{
    $('#ring').style.setProperty('--p',pct);
    document.querySelectorAll('.fill').forEach(el=>el.style.width=el.dataset.w+'%');
  });
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
