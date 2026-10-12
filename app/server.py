import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.seed import seed
        seed()
    except Exception as e:
        print(f"Seed skipped: {e}")
    yield


from app.agent import run_agent
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

app = FastAPI(title="Schedule RAG Agent", version="2.0", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentInput(BaseModel):
    input: str


class AgentOutput(BaseModel):
    output: str


@app.post("/agent/invoke", response_model=AgentOutput)
def agent_invoke(payload: AgentInput) -> AgentOutput:
    try:
        return AgentOutput(output=run_agent(payload.input))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def list_events(category: str = "all"):
    """Fetch all events from Pinecone for the next 30 days, optionally filtered by category."""
    from pinecone import Pinecone
    from app.tools import _embed

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    today = datetime.now().date()
    end_date = today + timedelta(days=30)
    today_str = today.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        result = index.query(
            vector=_embed("schedule events"),
            top_k=100,
            filter={"date": {"$gte": today_str, "$lte": end_str}},
            include_metadata=True,
        )
        matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
    except Exception:
        # fallback: no date filter
        result = index.query(
            vector=_embed("schedule events"),
            top_k=100,
            include_metadata=True,
        )
        matches = result.matches if hasattr(result, "matches") else result.get("matches", [])

    events = []
    seen = set()
    for m in matches:
        mid = m.id if hasattr(m, "id") else m.get("id", "")
        if mid in seen:
            continue
        seen.add(mid)
        meta = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
        cat = meta.get("event_type", "personal")
        if category != "all" and cat != category:
            continue
        events.append({
            "id": mid,
            "date": meta.get("date", ""),
            "time": meta.get("time", ""),
            "title": meta.get("title", meta.get("purpose", "")),
            "category": cat,
            "description": meta.get("description", ""),
        })

    events.sort(key=lambda x: (x["date"], x["time"]))
    return JSONResponse(content={"events": events})


class EventCreate(BaseModel):
    title: str
    date: str
    time: str = "09:00"
    event_type: str = "meeting"
    description: str = ""


@app.post("/events")
def create_event(payload: EventCreate):
    """Create an event directly in Pinecone — no LLM involved."""
    from app.tools import update_schedule
    try:
        result = update_schedule.invoke({
            "action": "add",
            "title": payload.title,
            "date": payload.date,
            "time": payload.time,
            "event_type": payload.event_type,
            "description": payload.description,
        })
        return {"status": "created", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    """Delete a specific event by ID directly from Pinecone."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(ids=[event_id])
        return {"status": "deleted", "id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/agent/playground", response_class=HTMLResponse)
def playground():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Schedule Assistant</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#f0f2ff;min-height:100vh;color:#1a1a2e}
.app{display:grid;grid-template-columns:230px 1fr;min-height:100vh}

/* SIDEBAR */
.sidebar{background:#1e2147;padding:0;display:flex;flex-direction:column;overflow-y:auto}
.logo{padding:20px 18px 18px;font-size:1rem;font-weight:700;color:#fff;border-bottom:1px solid #2d3170;display:flex;align-items:center;gap:8px}
.ns{padding:14px 16px 4px;font-size:.68rem;font-weight:700;color:#6b7db3;letter-spacing:.08em;text-transform:uppercase}
.nav-btn{display:flex;align-items:center;gap:9px;width:calc(100%-12px);margin:2px 6px;padding:9px 14px;background:none;border:none;color:#b0bce8;font-size:.85rem;cursor:pointer;border-radius:8px;text-align:left;transition:all .15s;width:calc(100% - 12px)}
.nav-btn:hover{background:#2d3170;color:#fff}
.nav-btn.active{background:#4f6ef7;color:#fff}
.cat-btn{display:flex;align-items:center;gap:9px;margin:1px 6px;padding:7px 14px;background:none;border:none;color:#b0bce8;font-size:.82rem;cursor:pointer;border-radius:8px;text-align:left;transition:all .15s;width:calc(100% - 12px)}
.cat-btn:hover{background:#2d3170;color:#fff}
.cat-btn.active{background:rgba(79,110,247,.28);color:#9ab3ff}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

/* MAIN */
.main{display:flex;flex-direction:column;overflow:hidden}
.topbar{background:#fff;padding:14px 26px;border-bottom:1px solid #e8eaf6;display:flex;align-items:center;justify-content:space-between;gap:12px}
.topbar h1{font-size:1.05rem;font-weight:700;color:#1a1a2e}
.topbar .today{font-size:.8rem;color:#888}
.content{padding:22px 26px;overflow-y:auto;flex:1}

/* EVENT CARDS */
.group-label{font-size:.75rem;font-weight:700;color:#888;letter-spacing:.06em;text-transform:uppercase;margin:18px 0 8px}
.group-label:first-child{margin-top:0}
.evt-list{display:flex;flex-direction:column;gap:7px;margin-bottom:4px}
.evt-card{background:#fff;border-radius:11px;padding:13px 15px;display:flex;align-items:center;gap:13px;box-shadow:0 1px 4px rgba(0,0,0,.06);transition:box-shadow .15s}
.evt-card:hover{box-shadow:0 3px 14px rgba(0,0,0,.1)}
.evt-accent{width:4px;height:44px;border-radius:4px;flex-shrink:0}
.evt-body{flex:1;min-width:0}
.evt-title{font-weight:600;font-size:.9rem;color:#1a1a2e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.evt-sub{font-size:.76rem;color:#888;margin-top:2px}
.evt-badge{font-size:.68rem;padding:3px 8px;border-radius:20px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap;flex-shrink:0}
.del-btn{background:none;border:none;color:#d1d5db;cursor:pointer;font-size:.95rem;padding:5px;border-radius:6px;transition:all .15s;flex-shrink:0}
.del-btn:hover{color:#ef4444;background:#fff1f2}
.empty{text-align:center;padding:56px 0;color:#aaa;font-size:.88rem}
.loading{text-align:center;padding:40px 0;color:#888;font-size:.88rem}

/* FORM */
.panel{background:#fff;border-radius:14px;padding:22px 24px;box-shadow:0 2px 12px rgba(0,0,0,.07);max-width:500px}
.panel h2{font-size:.95rem;font-weight:700;margin-bottom:18px;color:#1a1a2e}
label{display:block;font-size:.76rem;font-weight:600;color:#555;margin-bottom:4px;margin-top:13px}
label:first-of-type{margin-top:0}
input,select,textarea{width:100%;padding:9px 12px;border:1.5px solid #e0e4f0;border-radius:8px;font-size:.88rem;color:#1a1a2e;outline:none;font-family:inherit;transition:border .15s}
input:focus,select:focus,textarea:focus{border-color:#4f6ef7}
textarea{min-height:60px;resize:vertical}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:11px}
.btn{margin-top:16px;width:100%;padding:10px;background:#4f6ef7;color:#fff;border:none;border-radius:9px;font-size:.9rem;font-weight:600;cursor:pointer;transition:background .15s}
.btn:hover:not(:disabled){background:#3a57e8}
.btn:disabled{background:#b0bcf7;cursor:not-allowed}
.ps{margin-top:9px;font-size:.8rem;text-align:center;min-height:14px;color:#4f6ef7}
.ps.err{color:#ef4444}

/* ASK */
.ask-ans{margin-top:14px;background:#f5f7ff;border:1.5px solid #dde1f0;border-radius:10px;padding:13px 15px;font-size:.87rem;line-height:1.7;display:none;white-space:pre-wrap;color:#1a1a2e}
.ask-err{margin-top:7px;font-size:.8rem;color:#ef4444;min-height:14px}
</style>
</head>
<body>
<div class="app">

<div class="sidebar">
  <div class="logo">📅 Schedule Assistant</div>

  <div class="ns">Views</div>
  <button class="nav-btn active" onclick="showView('all')" id="nav-all">🗓️ &nbsp;All Events</button>
  <button class="nav-btn" onclick="showView('add')" id="nav-add">➕ &nbsp;Add Event</button>
  <button class="nav-btn" onclick="showView('ask')" id="nav-ask">🔍 &nbsp;Ask Schedule</button>

  <div class="ns" style="margin-top:10px">Categories</div>
  <button class="cat-btn active" onclick="filterCat('all')" id="cat-all"><span class="dot" style="background:#4f6ef7"></span>All</button>
  <button class="cat-btn" onclick="filterCat('education')" id="cat-education"><span class="dot" style="background:#0ea5e9"></span>Education</button>
  <button class="cat-btn" onclick="filterCat('work')" id="cat-work"><span class="dot" style="background:#6366f1"></span>Work</button>
  <button class="cat-btn" onclick="filterCat('health')" id="cat-health"><span class="dot" style="background:#22c55e"></span>Health</button>
  <button class="cat-btn" onclick="filterCat('family')" id="cat-family"><span class="dot" style="background:#f59e0b"></span>Family</button>
  <button class="cat-btn" onclick="filterCat('social')" id="cat-social"><span class="dot" style="background:#ec4899"></span>Social</button>
  <button class="cat-btn" onclick="filterCat('personal')" id="cat-personal"><span class="dot" style="background:#8b5cf6"></span>Personal</button>
  <button class="cat-btn" onclick="filterCat('shopping')" id="cat-shopping"><span class="dot" style="background:#f97316"></span>Shopping & Errands</button>
  <button class="cat-btn" onclick="filterCat('entertainment')" id="cat-entertainment"><span class="dot" style="background:#a855f7"></span>Entertainment</button>
  <button class="cat-btn" onclick="filterCat('travel')" id="cat-travel"><span class="dot" style="background:#ef4444"></span>Travel</button>
  <button class="cat-btn" onclick="filterCat('finance')" id="cat-finance"><span class="dot" style="background:#14b8a6"></span>Finance</button>
  <button class="cat-btn" onclick="filterCat('home')" id="cat-home"><span class="dot" style="background:#84cc16"></span>Home</button>
  <button class="cat-btn" onclick="filterCat('deadline')" id="cat-deadline"><span class="dot" style="background:#ef4444"></span>Deadlines</button>
  <button class="cat-btn" onclick="filterCat('recurring')" id="cat-recurring"><span class="dot" style="background:#06b6d4"></span>Recurring Events</button>
  <button class="cat-btn" onclick="filterCat('appointment')" id="cat-appointment"><span class="dot" style="background:#d946ef"></span>Appointments</button>
  <button class="cat-btn" onclick="filterCat('emergency')" id="cat-emergency"><span class="dot" style="background:#dc2626"></span>Emergencies</button>
  <button class="cat-btn" onclick="filterCat('meeting')" id="cat-meeting"><span class="dot" style="background:#3b82f6"></span>Meetings</button>
  <button class="cat-btn" onclick="filterCat('task')" id="cat-task"><span class="dot" style="background:#f59e0b"></span>Tasks</button>
  <button class="cat-btn" onclick="filterCat('workshop')" id="cat-workshop"><span class="dot" style="background:#10b981"></span>Workshops</button>
</div>

<div class="main">
  <div class="topbar">
    <h1 id="view-title">All Scheduled Events</h1>
    <span class="today" id="today-label"></span>
  </div>
  <div class="content">

    <div id="view-all">
      <div id="events-out"><div class="loading">Loading your schedule...</div></div>
    </div>

    <div id="view-add" style="display:none">
      <div class="panel">
        <h2>Manage Event</h2>
        <label>Action</label>
        <select id="f-action">
          <option value="add">Add new event</option>
          <option value="update">Update existing event</option>
          <option value="delete">Delete event</option>
        </select>
        <div class="row2">
          <div><label>Date</label><input type="date" id="f-date"/></div>
          <div><label>Time</label><input type="time" id="f-time" value="09:00"/></div>
        </div>
        <label>Title / Purpose</label>
        <input type="text" id="f-title" placeholder="e.g. Doctor visit, Grocery run, Team meeting"/>
        <label>Category</label>
        <select id="f-type">
          <option value="education">Education</option>
          <option value="work">Work</option>
          <option value="health">Health</option>
          <option value="family">Family</option>
          <option value="social">Social</option>
          <option value="personal">Personal</option>
          <option value="shopping">Shopping & Errands</option>
          <option value="entertainment">Entertainment</option>
          <option value="travel">Travel</option>
          <option value="finance">Finance</option>
          <option value="home">Home</option>
          <option value="deadline">Deadline</option>
          <option value="recurring">Recurring Event</option>
          <option value="appointment">Appointment</option>
          <option value="emergency">Emergency</option>
          <option value="meeting">Meeting</option>
          <option value="task">Task</option>
          <option value="workshop">Workshop</option>
        </select>
        <label>Description (optional)</label>
        <input type="text" id="f-desc" placeholder="Extra details..."/>
        <button class="btn" id="save-btn" onclick="saveEvent()">Save</button>
        <div class="ps" id="save-st"></div>
      </div>
    </div>

    <div id="view-ask" style="display:none">
      <div class="panel" style="max-width:560px">
        <h2>Ask about your schedule</h2>
        <label>Your question</label>
        <textarea id="q-in" rows="3" placeholder="e.g. What do I have tomorrow?&#10;Am I free Friday afternoon?&#10;Move my 2 PM meeting to 4 PM"></textarea>
        <button class="btn" id="ask-btn" onclick="askAgent()">Ask</button>
        <div class="ask-ans" id="ask-ans"></div>
        <div class="ask-err" id="ask-err"></div>
      </div>
    </div>

  </div>
</div>
</div>

<script>
const COLORS = {
  education:'#0ea5e9', work:'#6366f1', health:'#22c55e', family:'#f59e0b',
  social:'#ec4899', personal:'#8b5cf6', shopping:'#f97316', entertainment:'#a855f7',
  travel:'#ef4444', finance:'#14b8a6', home:'#84cc16', deadline:'#ef4444',
  recurring:'#06b6d4', appointment:'#d946ef', emergency:'#dc2626',
  meeting:'#3b82f6', task:'#f59e0b', workshop:'#10b981',
};
const BG = {};
Object.keys(COLORS).forEach(k => BG[k] = COLORS[k]+'22');

let allEvents = [];
let currentCat = 'all';

const today = new Date();
document.getElementById('today-label').textContent = today.toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'});
document.getElementById('f-date').value = today.toISOString().split('T')[0];

// ── VIEWS ──
function showView(name) {
  ['all','add','ask'].forEach(v => {
    document.getElementById('view-'+v).style.display = v===name ? '' : 'none';
    document.getElementById('nav-'+v).classList.toggle('active', v===name);
  });
  const T = {all:'All Scheduled Events', add:'Manage Event', ask:'Ask Schedule Assistant'};
  document.getElementById('view-title').textContent = T[name];
  if (name==='all') loadEvents();
}

// ── CATEGORY FILTER ──
function filterCat(cat) {
  currentCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('cat-'+cat).classList.add('active');
  showView('all');
}

// ── LOAD EVENTS (direct API) ──
async function loadEvents() {
  document.getElementById('events-out').innerHTML = '<div class="loading">Loading...</div>';
  try {
    const url = currentCat === 'all' ? '/events' : '/events?category='+currentCat;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to load');
    allEvents = data.events || [];
    renderEvents();
  } catch(e) {
    document.getElementById('events-out').innerHTML = '<div class="empty">Could not load events: '+e.message+'</div>';
  }
}

function renderEvents() {
  const out = document.getElementById('events-out');
  if (!allEvents.length) {
    out.innerHTML = '<div class="empty">No events found' + (currentCat!=='all' ? ' in this category' : '') + '.</div>';
    return;
  }
  const grouped = {};
  allEvents.forEach(e => { if (!grouped[e.date]) grouped[e.date]=[]; grouped[e.date].push(e); });

  let html = '';
  Object.keys(grouped).sort().forEach(date => {
    html += '<div class="group-label">'+fmtDate(date)+'</div><div class="evt-list">';
    grouped[date].forEach(e => {
      const cat = e.category || 'personal';
      const c = COLORS[cat] || '#4f6ef7';
      const bg = BG[cat] || '#eef2ff';
      const label = cat.charAt(0).toUpperCase()+cat.slice(1).replace('_',' ');
      html += `<div class="evt-card">
        <div class="evt-accent" style="background:${c}"></div>
        <div class="evt-body">
          <div class="evt-title">${e.title||'(no title)'}</div>
          <div class="evt-sub">${e.time||''}${e.description ? ' &nbsp;·&nbsp; '+e.description : ''}</div>
        </div>
        <span class="evt-badge" style="background:${bg};color:${c}">${label}</span>
        <button class="del-btn" title="Delete" onclick="delEvent('${e.id}','${(e.title||'').replace(/'/g,"\\'")}')">🗑️</button>
      </div>`;
    });
    html += '</div>';
  });
  out.innerHTML = html;
}

function fmtDate(ds) {
  const d = new Date(ds+'T12:00:00');
  const ts = today.toISOString().split('T')[0];
  const tm = new Date(today.getTime()+86400000).toISOString().split('T')[0];
  const fmt = d.toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric',year:'numeric'});
  if (ds===ts) return '📌 Today — '+fmt;
  if (ds===tm) return '📅 Tomorrow — '+fmt;
  return fmt;
}

// ── DELETE (direct API) ──
async function delEvent(id, title) {
  if (!confirm('Delete "'+title+'"?')) return;
  try {
    const res = await fetch('/events/'+encodeURIComponent(id), {method:'DELETE'});
    if (!res.ok) { const d=await res.json(); throw new Error(d.detail); }
    allEvents = allEvents.filter(e => e.id !== id);
    renderEvents();
  } catch(e) { alert('Delete failed: '+e.message); }
}

// ── SAVE EVENT ──
async function saveEvent() {
  const action = document.getElementById('f-action').value;
  const date   = document.getElementById('f-date').value;
  const time   = document.getElementById('f-time').value || '09:00';
  const title  = document.getElementById('f-title').value.trim();
  const type   = document.getElementById('f-type').value;
  const desc   = document.getElementById('f-desc').value.trim();
  const btn    = document.getElementById('save-btn');
  const st     = document.getElementById('save-st');

  if (!date || !title) { st.className='ps err'; st.textContent='Date and title are required.'; return; }
  btn.disabled=true; st.className='ps'; st.textContent='Saving...';

  try {
    if (action === 'add') {
      // Direct Pinecone save — no LLM
      const res = await fetch('/events', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title, date, time, event_type: type, description: desc}),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      st.textContent = '✓ Saved!';
      document.getElementById('f-title').value = '';
      document.getElementById('f-desc').value = '';
    } else {
      // Update/delete still uses agent
      const msg = action==='delete'
        ? `Delete the event titled "${title}" on ${date}`
        : `Update event "${title}" on ${date}: change time to ${time}, category ${type}. ${desc}`;
      const res = await fetch('/agent/invoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:msg})});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      st.textContent = action==='delete' ? '✓ Deleted.' : '✓ Updated.';
    }
  } catch(e) { st.className='ps err'; st.textContent='Error: '+e.message; }
  finally { btn.disabled=false; }
}

// ── ASK AGENT ──
async function askAgent() {
  const q = document.getElementById('q-in').value.trim();
  const btn = document.getElementById('ask-btn');
  const ans = document.getElementById('ask-ans');
  const err = document.getElementById('ask-err');
  if (!q) return;
  btn.disabled=true; btn.textContent='Thinking...'; ans.style.display='none'; err.textContent='';
  try {
    const res = await fetch('/agent/invoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:q})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    ans.style.display='block'; ans.textContent=data.output;
  } catch(e) { err.textContent='Error: '+e.message; }
  finally { btn.disabled=false; btn.textContent='Ask'; }
}

document.getElementById('q-in').addEventListener('keydown',e=>{ if(e.key==='Enter'&&e.ctrlKey) askAgent(); });
loadEvents();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
