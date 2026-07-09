#!/usr/bin/env python3
"""
Manual annotation website for the real-robot demos. Two tabs:
  1. Segments (video): scrub the MuJoCo replay, drag start/end markers, add segments.
  2. Events (plot):     for each segment, drag a marker on its signal plot to the disturbance
                        onset (event). Start is usually earlier than the event; the event is what
                        t_unsafe / temporal-recall are measured from.

annotations.json = { demo: {"task", "segments": [[s,e]s,...], "events": [ev_s, ...]} }
Video/plot time (s) == demo time == data_index/50.

Run (stdlib only):  python safety_value/scripts/1_data_prep/annotate_server.py --port 8000
"""
import argparse, json, os, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLAY_DIR = "logs/real_exp_replay"
ANN_FILE = os.path.join(REPLAY_DIR, "annotations.json")
DEMOS = [
    ("demo_push_1", "push"), ("demo_push_2", "push"), ("demo_push_3", "push"), ("demo_push_4", "push"),
    ("push_1_front", "push"), ("push_2_front", "push"), ("push_3_front", "push"), ("push_4_front", "push"),
    ("push_5_left", "push"), ("push_6_left", "push"), ("push_7_left", "push"), ("push_8_left", "push"),
    ("demo_avoid_1", "avoid"), ("demo_avoid_2", "avoid"),
]

def load_ann():
    if os.path.exists(ANN_FILE):
        try: return json.load(open(ANN_FILE))
        except Exception: return {}
    return {}

def save_ann(ann):
    os.makedirs(REPLAY_DIR, exist_ok=True); json.dump(ann, open(ANN_FILE, "w"), indent=2)

def demo_duration(demo):
    meta = os.path.join(REPLAY_DIR, demo + ".meta")
    if os.path.exists(meta):
        try: return float(open(meta).read().splitlines()[1])
        except Exception: return None
    return None

def build_manifest():
    ann = load_ann()
    return [{"demo": d, "task": t, "rendered": os.path.exists(os.path.join(REPLAY_DIR, d + ".mp4")),
             "duration": demo_duration(d), "segments": ann.get(d, {}).get("segments", []),
             "events": ann.get(d, {}).get("events", [])} for d, t in DEMOS]

def build_segsignals():
    order = {d: i for i, (d, _) in enumerate(DEMOS)}; out = []
    for task in ("push", "avoid"):
        fp = os.path.join(REPLAY_DIR, "seg_signals_%s.json" % task)
        if os.path.exists(fp):
            try: out += json.load(open(fp)).get("segments", [])
            except Exception: pass
    out.sort(key=lambda s: (order.get(s.get("demo"), 99), s.get("k", 0)))
    return out

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Annotator</title><style>
 html,body{height:100%;margin:0;overflow:hidden}
 body{font-family:system-ui,Arial;background:#111;color:#eee;display:flex;flex-direction:column}
 #tabs{display:flex;gap:6px;padding:6px 10px;background:#000;border-bottom:1px solid #333;flex:0 0 auto}
 .tabb{font-size:14px;padding:6px 14px;background:#222;color:#bbb;border:1px solid #444;border-radius:6px 6px 0 0;cursor:pointer}
 .tabb.active{background:#2557a7;color:#fff}
 select,button{font-size:13px;padding:4px 8px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;cursor:pointer}
 button:hover{background:#3a3a3a}
 .bar{padding:6px 12px;background:#1b1b1b;border-bottom:1px solid #333;display:flex;gap:10px;align-items:center;flex-wrap:wrap;flex:0 0 auto}
 .badge{padding:2px 8px;border-radius:10px;font-size:12px}.push{background:#2557a7}.avoid{background:#9a3b1f}
 .view{flex:1 1 auto;min-height:0;display:none;flex-direction:column;padding:8px 12px;max-width:1100px;width:100%;margin:0 auto;box-sizing:border-box}
 video{flex:0 1 auto;min-height:0;max-height:50vh;width:auto;max-width:100%;background:#000;border-radius:6px;display:block;margin:0 auto}
 #track{position:relative;height:40px;background:#222;border-radius:6px;margin:8px 0;cursor:pointer;user-select:none;flex:0 0 auto}
 .seg{position:absolute;top:0;height:100%;background:rgba(80,200,120,.30);border-left:1px solid #5c8;border-right:1px solid #5c8}
 #playhead{position:absolute;top:0;width:2px;height:100%;background:#fff}
 .handle{position:absolute;top:-4px;width:12px;height:48px;border-radius:3px;cursor:ew-resize}
 #hs{background:#3cba54}#he{background:#e0453e}
 #seglist{margin-top:8px;flex:1 1 auto;min-height:0;overflow-y:auto}
 .row{display:flex;gap:10px;align-items:center;padding:3px 0;border-bottom:1px solid #2a2a2a}
 .mono{font-family:monospace}#hint,#ehint{font-size:12px;color:#9a9}#save{background:#2e7d32}
 #ecanvas{flex:1 1 auto;min-height:0;width:100%;background:#181818;border-radius:6px;cursor:crosshair}
 #ctrls{flex:0 0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
</style></head><body>
<div id=tabs>
  <div class="tabb active" id=tabSeg>1 · Segments (video)</div>
  <div class=tabb id=tabEvt>2 · Events (plot)</div>
  <span id=gstatus style="color:#8c8;align-self:center;font-size:12px"></span>
</div>

<!-- ============ SEGMENTS TAB ============ -->
<div id=segview class=view style="display:flex">
  <div class=bar>
    <select id=demosel></select><span id=task class=badge></span><span id=prog></span>
    <button id=prev>◀ Prev</button><button id=next>Next ▶</button><button id=refresh>↻ Refresh</button>
    <span id=hint>scrub • [ start • ] end • Enter add • drag green/red handles</span>
  </div>
  <video id=vid controls preload=auto></video>
  <div id=track><div id=playhead></div><div id=hs class=handle></div><div id=he class=handle></div></div>
  <div id=ctrls>
    <button id=setstart>[ start=playhead</button><span class=mono id=startv>0.00</span>
    <button id=setend>] end=playhead</button><span class=mono id=endv>0.00</span>
    <button id=add>＋ Add</button><button id=save>💾 Save</button><span id=status style="color:#8c8"></span>
  </div>
  <div id=seglist></div>
</div>

<!-- ============ EVENTS TAB ============ -->
<div id=evtview class=view>
  <div class=bar>
    <span id=eprog></span><span id=etask class=badge></span>
    <button id=eprev>◀ Prev seg</button><button id=enext>Seg ▶</button>
    <button id=eunmarked>Next unmarked ▶</button>
    <button id=esetstart>event = start</button><span class=mono id=eev></span>
    <span id=ehint>click/drag on plot to set EVENT (disturbance onset) • ←/→ prev/next • autosaved</span>
  </div>
  <canvas id=ecanvas></canvas>
  <div style="flex:0 0 auto;font-size:12px;color:#aaa;padding-top:4px">
    white = safety_signal(GT) &nbsp; magenta = value(logged) &nbsp; cyan = ball dist (avoid) &nbsp; orange = EVENT &nbsp; red dashed = 0
  </div>
</div>
<script>
let MAN=[], cur=0, dur=1, segs=[], startT=0, endT=0, dragging=null;
const vid=document.getElementById('vid'), track=document.getElementById('track');
const hs=document.getElementById('hs'), he=document.getElementById('he'), ph=document.getElementById('playhead');
const fmt=t=>t.toFixed(2)+'s', pct=t=>Math.max(0,Math.min(100,t/dur*100));
function gstat(m){document.getElementById('gstatus').textContent=m;}
async function loadManifest(){MAN=await (await fetch('/api/manifest')).json();
  const sel=document.getElementById('demosel');sel.innerHTML='';
  MAN.forEach((m,i)=>{const o=document.createElement('option');o.value=i;
    o.textContent=(m.rendered?'':'⏳ ')+m.demo+(m.segments.length?(' ('+m.segments.length+')'):'');sel.appendChild(o);});
  sel.value=cur;}
function loadDemo(i){cur=i;const m=MAN[i];document.getElementById('demosel').value=i;
  document.getElementById('task').textContent=m.task;document.getElementById('task').className='badge '+m.task;
  document.getElementById('prog').textContent='Demo '+(i+1)+' / '+MAN.length;
  dur=m.duration||1;segs=JSON.parse(JSON.stringify(m.segments||[]));startT=0;endT=dur;
  document.getElementById('status').textContent=m.rendered?'':'rendering — Refresh later';
  vid.src='/video/'+m.demo+'.mp4?'+Date.now();vid.load();renderSeg();}
function renderSeg(){document.getElementById('startv').textContent=fmt(startT);document.getElementById('endv').textContent=fmt(endT);
  hs.style.left='calc('+pct(startT)+'% - 6px)';he.style.left='calc('+pct(endT)+'% - 6px)';
  [...track.querySelectorAll('.seg')].forEach(e=>e.remove());
  segs.forEach(s=>{const d=document.createElement('div');d.className='seg';d.style.left=pct(s[0])+'%';d.style.width=(pct(s[1])-pct(s[0]))+'%';track.appendChild(d);});
  const sl=document.getElementById('seglist');sl.innerHTML='<b>Segments ('+segs.length+')</b>';
  segs.forEach((s,j)=>{const r=document.createElement('div');r.className='row';
    r.innerHTML='<span class=mono>#'+j+'  ['+s[0].toFixed(2)+', '+s[1].toFixed(2)+']s  ('+(s[1]-s[0]).toFixed(2)+'s)</span>';
    const g=document.createElement('button');g.textContent='go';g.onclick=()=>vid.currentTime=s[0];
    const b=document.createElement('button');b.textContent='del';b.onclick=()=>{segs.splice(j,1);renderSeg();};
    r.appendChild(g);r.appendChild(b);sl.appendChild(r);});}
vid.addEventListener('timeupdate',()=>ph.style.left=pct(vid.currentTime)+'%');
vid.addEventListener('loadedmetadata',()=>{if(!MAN[cur].duration){dur=vid.duration;endT=dur;renderSeg();}});
const tFromX=x=>{const r=track.getBoundingClientRect();return Math.max(0,Math.min(dur,(x-r.left)/r.width*dur));};
track.addEventListener('mousedown',e=>{if(e.target===hs){dragging='s';return;}if(e.target===he){dragging='e';return;}vid.currentTime=tFromX(e.clientX);});
document.addEventListener('mousemove',e=>{if(!dragging)return;const t=tFromX(e.clientX);if(dragging==='s')startT=t;else endT=t;renderSeg();});
document.addEventListener('mouseup',()=>dragging=null);
document.getElementById('setstart').onclick=()=>{startT=vid.currentTime;renderSeg();};
document.getElementById('setend').onclick=()=>{endT=vid.currentTime;renderSeg();};
document.getElementById('add').onclick=()=>{let a=Math.min(startT,endT),b=Math.max(startT,endT);if(b-a<0.05){alert('too short');return;}segs.push([+a.toFixed(2),+b.toFixed(2)]);segs.sort((x,y)=>x[0]-y[0]);renderSeg();};
async function saveSeg(){const m=MAN[cur];await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({demo:m.demo,task:m.task,segments:segs})});m.segments=JSON.parse(JSON.stringify(segs));document.getElementById('status').textContent='saved '+segs.length+' ✓';loadManifest();}
document.getElementById('save').onclick=saveSeg;
document.getElementById('prev').onclick=async()=>{await saveSeg();if(cur>0)loadDemo(cur-1);};
document.getElementById('next').onclick=async()=>{await saveSeg();if(cur<MAN.length-1)loadDemo(cur+1);};
document.getElementById('demosel').onchange=async e=>{await saveSeg();loadDemo(+e.target.value);};
document.getElementById('refresh').onclick=async()=>{await loadManifest();loadDemo(cur);};

/* ===== EVENTS TAB ===== */
let SS=[], ei=0, evMap={}, ecv=document.getElementById('ecanvas'), ectx=ecv.getContext('2d');
function buildEvMap(){evMap={};MAN.forEach(m=>{const n=(m.segments||[]).length;let e=m.events||[];if(e.length!==n)e=(m.segments||[]).map(()=>null);evMap[m.demo]=e.slice();});}
async function ensureSS(){SS=await (await fetch('/api/segsignals')).json();buildEvMap();}
function curEv(){const s=SS[ei];const e=evMap[s.demo][s.k];return (e==null)?s.start:e;}
async function saveEvtDemo(demo){const m=MAN.find(x=>x.demo===demo);if(!m)return;
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({demo:demo,task:m.task,segments:m.segments,events:evMap[demo]})});
  m.events=evMap[demo].slice();}
function setEv(t){const s=SS[ei];t=Math.max(s.start,Math.min(s.end,t));evMap[s.demo][s.k]=+t.toFixed(2);drawEvt();}
function drawEvt(){const s=SS[ei];if(!s){return;}
  document.getElementById('etask').textContent=s.task;document.getElementById('etask').className='badge '+s.task;
  const nMarked=SS.filter(x=>evMap[x.demo][x.k]!=null).length;
  document.getElementById('eprog').textContent='Segment '+(ei+1)+' / '+SS.length+'  ('+nMarked+' marked)  —  '+s.name+'  ['+s.start+','+s.end+']s';
  document.getElementById('eev').textContent='event = '+curEv().toFixed(2)+'s'+(evMap[s.demo][s.k]==null?' (unset→start)':'');
  const W=ecv.width=ecv.clientWidth, H=ecv.height=ecv.clientHeight;
  ectx.clearRect(0,0,W,H);
  const mL=42,mR=44,mT=10,mB=22,pw=W-mL-mR,ph2=H-mT-mB,x0=s.start,x1=s.end;
  const xt=t=>mL+(t-x0)/(x1-x0||1)*pw, ys=v=>mT+(1.4-v)/(2.6)*ph2;
  // unsafe fill
  ectx.fillStyle='rgba(220,60,60,.18)';
  for(let j=0;j<s.l.length;j++){if(s.l[j]>0){const xa=xt(s.t[j]),xb=xt(s.t[Math.min(j+1,s.t.length-1)]);ectx.fillRect(xa,mT,Math.max(1,xb-xa),ph2);}}
  // zero line
  ectx.strokeStyle='#a33';ectx.setLineDash([4,4]);ectx.beginPath();ectx.moveTo(mL,ys(0));ectx.lineTo(W-mR,ys(0));ectx.stroke();ectx.setLineDash([]);
  ectx.fillStyle='#777';ectx.font='10px monospace';[-1,0,1].forEach(v=>ectx.fillText(''+v,6,ys(v)+3));
  const line=(arr,col,lw)=>{if(!arr)return;ectx.strokeStyle=col;ectx.lineWidth=lw;ectx.beginPath();arr.forEach((v,j)=>{const xx=xt(s.t[j]),yy=ys(v);j?ectx.lineTo(xx,yy):ectx.moveTo(xx,yy);});ectx.stroke();};
  if(s.d){const dmax=Math.max(...s.d,0.6),yd=v=>mT+(1-v/dmax)*ph2;
    ectx.strokeStyle='#36c6c6';ectx.lineWidth=1;ectx.beginPath();s.d.forEach((v,j)=>{const xx=xt(s.t[j]),yy=yd(v);j?ectx.lineTo(xx,yy):ectx.moveTo(xx,yy);});ectx.stroke();
    ectx.setLineDash([2,3]);ectx.beginPath();ectx.moveTo(mL,yd(0.5));ectx.lineTo(W-mR,yd(0.5));ectx.stroke();ectx.setLineDash([]);}
  line(s.v,'#e060e0',1.3); line(s.l,'#ffffff',2);
  // event line
  const ex=xt(curEv());ectx.strokeStyle='#ff9800';ectx.lineWidth=3;ectx.beginPath();ectx.moveTo(ex,0);ectx.lineTo(ex,H);ectx.stroke();
  ectx.fillStyle='#ff9800';ectx.fillText('EVENT',ex+4,mT+10);}
function evtFromX(x){const s=SS[ei],r=ecv.getBoundingClientRect(),mL=42,mR=44,pw=r.width-mL-mR;return s.start+(x-r.left-mL)/pw*(s.end-s.start);}
let eDrag=false;
ecv.addEventListener('mousedown',e=>{eDrag=true;setEv(evtFromX(e.clientX));});
ecv.addEventListener('mousemove',e=>{if(eDrag)setEv(evtFromX(e.clientX));});
document.addEventListener('mouseup',()=>{if(eDrag){eDrag=false;saveEvtDemo(SS[ei].demo).then(()=>gstat('event saved ✓'));}});
function gotoSeg(i){if(i<0||i>=SS.length)return;ei=i;drawEvt();}
document.getElementById('eprev').onclick=()=>gotoSeg(ei-1);
document.getElementById('enext').onclick=()=>gotoSeg(ei+1);
document.getElementById('esetstart').onclick=()=>{setEv(SS[ei].start);saveEvtDemo(SS[ei].demo);};
document.getElementById('eunmarked').onclick=()=>{for(let d=1;d<=SS.length;d++){const j=(ei+d)%SS.length;if(evMap[SS[j].demo][SS[j].k]==null){gotoSeg(j);return;}}gstat('all segments marked ✓');};

/* ===== tabs ===== */
const segview=document.getElementById('segview'), evtview=document.getElementById('evtview');
document.getElementById('tabSeg').onclick=()=>{segview.style.display='flex';evtview.style.display='none';tabSeg.classList.add('active');tabEvt.classList.remove('active');};
document.getElementById('tabEvt').onclick=async()=>{await saveSeg();await ensureSS();segview.style.display='none';evtview.style.display='flex';tabEvt.classList.add('active');tabSeg.classList.remove('active');if(ei>=SS.length)ei=0;drawEvt();};
window.addEventListener('resize',()=>{if(evtview.style.display==='flex')drawEvt();});
document.addEventListener('keydown',e=>{
  if(evtview.style.display==='flex'){if(e.key==='ArrowLeft'){gotoSeg(ei-1);}else if(e.key==='ArrowRight'){gotoSeg(ei+1);}return;}
  if(e.target.tagName==='INPUT')return;
  if(e.key==='[')document.getElementById('setstart').click();
  else if(e.key===']')document.getElementById('setend').click();
  else if(e.key==='Enter')document.getElementById('add').click();
  else if(e.key===' '){e.preventDefault();vid.paused?vid.play():vid.pause();}});
(async()=>{await loadManifest();loadDemo(0);})();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        try: self.wfile.write(body)
        except BrokenPipeError: pass
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode()); return
        if self.path == "/api/manifest":
            self._send(200, "application/json", json.dumps(build_manifest()).encode()); return
        if self.path == "/api/segsignals":
            self._send(200, "application/json", json.dumps(build_segsignals()).encode()); return
        if self.path.startswith("/video/"):
            fp = os.path.join(REPLAY_DIR, os.path.basename(self.path.split("?")[0][len("/video/"):]))
            if not os.path.exists(fp): self._send(404, "text/plain", b"not rendered yet"); return
            self.serve_range(fp, "video/mp4"); return
        self._send(404, "text/plain", b"404")
    def do_POST(self):
        if self.path == "/api/save":
            n = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(n) or b"{}")
            ann = load_ann(); entry = ann.get(data["demo"], {})
            if data.get("task") is not None: entry["task"] = data["task"]
            if "segments" in data: entry["segments"] = data["segments"]
            if "events" in data: entry["events"] = data["events"]
            ann[data["demo"]] = entry; save_ann(ann)
            self._send(200, "application/json", b'{"ok":true}'); return
        self._send(404, "text/plain", b"404")
    def serve_range(self, path, ctype):
        fs = os.path.getsize(path); rng = self.headers.get("Range")
        try:
            if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else fs - 1
                end = min(end, fs - 1); length = end - start + 1
                self.send_response(206); self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Range", f"bytes {start}-{end}/{fs}")
                self.send_header("Content-Length", str(length)); self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start); rem = length
                    while rem > 0:
                        c = f.read(min(1 << 20, rem))
                        if not c: break
                        self.wfile.write(c); rem -= len(c)
            else:
                self.send_response(200); self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(fs)); self.end_headers()
                with open(path, "rb") as f:
                    while True:
                        c = f.read(1 << 20)
                        if not c: break
                        self.wfile.write(c)
        except (BrokenPipeError, ConnectionResetError): pass

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); args = ap.parse_args()
    print("Annotation server on http://localhost:%d  (annotations -> %s)" % (args.port, ANN_FILE))
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()

if __name__ == "__main__":
    main()
