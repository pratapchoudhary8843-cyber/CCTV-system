
import os, time, json, traceback, sqlite3, threading, collections, webbrowser
from pathlib import Path
import cv2, numpy as np, requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

SOURCE = os.getenv("SOURCE", "auto")
MODEL = os.getenv("MODEL", "yolov8n.pt")
CONF = float(os.getenv("CONF", 0.35))
IMGSZ = int(os.getenv("IMGSZ", 960))                 
CROWD = int(os.getenv("CROWD_LIMIT", 5))              
WATCH = set(os.getenv("WATCH_OBJECTS", "backpack,suitcase,knife").split(","))
NOTIFY = set(os.getenv("NOTIFY", "intrusion,object,crowd").split(","))
PRE, POST, COOLDOWN = 3, 5, int(os.getenv("COOLDOWN", 30))
TG_TOKEN, TG_CHAT, WEBHOOK = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("WEBHOOK_URL")

os.makedirs("data/clips", exist_ok=True); os.makedirs("data/snaps", exist_ok=True)
db = sqlite3.connect("data/cctv.db", check_same_thread=False); dblock = threading.Lock()
db.executescript("""CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, type TEXT, label TEXT, snapshot TEXT, clip TEXT);
CREATE TABLE IF NOT EXISTS counts(ts INTEGER, label TEXT, n INTEGER);""")
ZONE_FILE = "data/zone.json"
ZONE = json.load(open(ZONE_FILE)) if os.path.exists(ZONE_FILE) else [[.25,.25],[.75,.25],[.75,.9],[.25,.9]]
state = {"jpeg": None, "t": 0, "counts": {}, "people_today": 0, "recording": False, "fps": 0, "status": "Starting…", "error": None, "source": None}
det = None

def q(sql, args=(), many=False, commit=False):
    with dblock:
        cur = db.executemany(sql, args) if many else db.execute(sql, args)
        rows = cur.fetchall()
        if commit: db.commit()
        return rows

def notify(kind, label, snap):
    if kind not in NOTIFY: return
    msg = f"ALERT: {kind} ({label}) at {time.strftime('%H:%M:%S')}"
    try:
        if TG_TOKEN:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto", data={"chat_id": TG_CHAT, "caption": msg},
                          files={"photo": open(snap, "rb")}, timeout=10)
        if WEBHOOK: requests.post(WEBHOOK, json={"type": kind, "label": label, "time": time.time(), "message": msg}, timeout=10)
    except Exception as e: print("notify failed:", e)

def open_writer(stamp, kind, fps, w, h):
    fps = int(round(max(5, min(30, fps))))
    for cc, ext in (("avc1", "mp4"), ("VP80", "webm"), ("mp4v", "mp4")):   
        path = f"data/clips/{stamp}_{kind}.{ext}"
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wr.isOpened(): return wr, path
        if os.path.exists(path): os.remove(path)
    return None, None

class Detector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(MODEL); self.lock = threading.Lock()
        self.pre = collections.deque(); self.writer = None; self.wsize = None; self.clip = None; self.rec_until = 0
        self.last = {}; self.minute = 0; self.bucket = collections.Counter()
        self.tracking = True; self.people = set(); self.day = time.strftime('%Y%m%d'); self.t_prev = None; self.fps = 15.0; self.n = 0; self.t0 = time.time()

    def infer(self, frame):
        if self.tracking:
            try: return self.model.track(frame, persist=True, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            except Exception as e:      
                self.tracking = False; print("Tracking unavailable, using plain detection:", e)
        return self.model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]

    def process(self, frame):
        with self.lock:
            h, w = frame.shape[:2]
            if w > 960: frame = cv2.resize(frame, (960, int(h * 960 / w))); h, w = frame.shape[:2]
            now = time.time()
            if self.t_prev: self.fps = .9 * self.fps + .1 * min(30.0, 1 / max(now - self.t_prev, 1e-3))
            self.t_prev = now
            zone = np.array([[x * w, y * h] for x, y in ZONE], np.int32)
            r = self.infer(frame)
            counts, intruders, watched = collections.Counter(), 0, set()
            for b in r.boxes:
                label = self.model.names[int(b.cls)]; counts[label] += 1
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                inside = label == "person" and cv2.pointPolygonTest(zone, ((x1 + x2) / 2, y2), False) >= 0
                intruders += inside
                if label == "person" and b.id is not None: self.people.add(int(b.id))
                if label in WATCH: watched.add(label)
                col = (60, 60, 230) if inside or label in WATCH else (160, 190, 90)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                tid = f"#{int(b.id)} " if b.id is not None else ""
                cv2.putText(frame, f"{tid}{label} {float(b.conf):.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 2)
            cv2.polylines(frame, [zone], True, (60, 170, 245), 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)

            if time.strftime('%Y%m%d') != self.day: self.people.clear(); self.day = time.strftime('%Y%m%d')
            npeople = counts["person"]; crowd = npeople >= CROWD
            cv2.putText(frame, f"People: {npeople}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (60, 60, 230) if crowd else (255, 255, 255), 2)
            events = [("intrusion", "person")] if intruders else []
            if crowd: events.append(("crowd", "person"))
            elif npeople and not intruders: events.append(("person", "person"))
            events += [("object", o) for o in watched]
            for kind, label in events:
                self.rec_until = now + POST
                if now - self.last.get((kind, label), 0) < COOLDOWN: continue
                self.last[(kind, label)] = now
                stamp = time.strftime("%Y%m%d_%H%M%S"); snap = f"data/snaps/{stamp}_{kind}.jpg"; cv2.imwrite(snap, frame)
                if self.writer is None:
                    self.writer, self.clip = open_writer(stamp, kind, self.fps, w, h); self.wsize = (w, h)
                    if self.writer:
                        for _, f in self.pre: self.writer.write(f)
                q("INSERT INTO events(ts,type,label,snapshot,clip) VALUES(?,?,?,?,?)", (now, kind, label, snap, self.clip), commit=True)
                threading.Thread(target=notify, args=(kind, label, snap), daemon=True).start()
            if self.writer:
                if self.wsize != (w, h): self.writer.release(); self.writer = None
                else:
                    self.writer.write(frame)
                    if now > self.rec_until: self.writer.release(); self.writer = None
            self.pre.append((now, frame.copy()))
            while self.pre and now - self.pre[0][0] > PRE: self.pre.popleft()

            m = int(now // 60 * 60)                      
            if m != self.minute and self.bucket:
                q("INSERT INTO counts VALUES(?,?,?)", [(self.minute, k, v) for k, v in self.bucket.items()], many=True, commit=True)
                self.bucket.clear()
            self.minute = m
            for k, v in counts.items(): self.bucket[k] = max(self.bucket[k], v)
            self.n += 1
            if now - self.t0 >= 1: state["fps"], self.n, self.t0 = self.n, 0, now
            state.update(jpeg=cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes(), t=now,
                         counts=dict(counts), people_today=len(self.people), recording=self.writer is not None)

def open_cam(src):
    if isinstance(src, int):                       
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for idx in [src] + [i for i in range(4) if i != src]:
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if cap.isOpened() and cap.read()[0]: print(f"Server camera opened: index {idx}"); return cap
                cap.release()
        return cv2.VideoCapture()
    return cv2.VideoCapture(src)

def camera_loop():
    if SOURCE == "browser":
        state["status"] = "Ready. Open the dashboard and click “Use this computer’s camera”."; return
    src = 0 if SOURCE == "auto" else int(SOURCE) if SOURCE.isdigit() else SOURCE
    while True:
        try:
            cap = open_cam(src)
            if not cap.isOpened():
                if SOURCE == "auto":
                    state["status"] = "No server camera found. Click “Use this computer’s camera” on the dashboard."; return
                state["error"] = f"Cannot open {src!r}. Check the address or file path, or close apps using the camera."
                time.sleep(5); continue
            state.update(error=None, source="server", status=f"Server camera {src!r} is running")
            while True:
                ok, frame = cap.read()
                if not ok: break
                det.process(frame)
            cap.release(); time.sleep(1)
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"; time.sleep(5)

def boot():
    global det
    try:
        state["status"] = "Loading AI model (first run downloads about 6 MB)…"
        det = Detector(); state["status"] = "Model ready"
        camera_loop()
    except Exception as e:
        state["error"] = f"Could not load the AI model ({type(e).__name__}: {e}). Check your internet connection and run again."

app = FastAPI()

@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "path": request.url.path}, status_code=500)
threading.Thread(target=boot, daemon=True).start()

@app.post("/api/frame")
async def frame(request: Request):                 
    if det is None: return JSONResponse({"ready": False, "status": state["status"]}, status_code=503)
    img = cv2.imdecode(np.frombuffer(await request.body(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return JSONResponse({"error": "bad image"}, status_code=400)
    state["source"] = "browser"; state["error"] = None
    try: await run_in_threadpool(det.process, img)
    except Exception as e:
        traceback.print_exc(); state["error"] = f"Detection error: {type(e).__name__}: {e}"
        return JSONResponse({"error": state["error"]}, status_code=500)
    return {"ready": True, "counts": state["counts"]}

@app.get("/video")
def video():
    def gen():
        while True:
            if state["jpeg"]: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + state["jpeg"] + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

@app.get("/api/live")
def live():
    d = {k: state[k] for k in ("counts", "people_today", "recording", "fps", "error", "status", "source")}
    return d | {"has_video": time.time() - state["t"] < 3, "model_ready": det is not None}

@app.get("/api/events")
def events(limit: int = 40):
    rows = q("SELECT id,ts,type,label,snapshot,clip FROM events ORDER BY ts DESC LIMIT ?", (limit,))
    return [dict(id=r[0], ts=r[1], type=r[2], label=r[3], snapshot="/media/" + r[4][5:], clip="/media/" + r[5][5:] if r[5] else None) for r in rows]

@app.get("/api/analytics")
def analytics():
    since = time.time() - 86400; today = time.mktime(time.localtime()[:3] + (0,) * 6)
    hr = "CAST(strftime('%H',ts,'unixepoch','localtime') AS INT)"
    by_type = {t: [0] * 24 for t in ("intrusion", "crowd", "person", "object")}
    for h, t, c in q(f"SELECT {hr},type,COUNT(*) FROM events WHERE ts>? GROUP BY 1,2", (since,)): by_type.setdefault(t, [0] * 24)[h] = c
    traffic = [0] * 24
    for h, a in q(f"SELECT {hr},AVG(n) FROM counts WHERE label='person' AND ts>? GROUP BY 1", (since,)): traffic[h] = round(a, 1)
    totals = dict(q("SELECT type,COUNT(*) FROM events WHERE ts>=? GROUP BY type", (today,)))
    per_hour = [sum(v[h] for v in by_type.values()) for h in range(24)]
    return dict(by_type=by_type, traffic=traffic, totals=totals, busiest=per_hour.index(max(per_hour)) if any(per_hour) else None)

@app.get("/api/zone")
def get_zone(): return ZONE

@app.post("/api/zone")
def set_zone(points: list[list[float]]):
    if len(points) >= 3: ZONE[:] = points; json.dump(ZONE, open(ZONE_FILE, "w"))
    return ZONE

STATIC = Path(__file__).resolve().parent / "static"

@app.get("/")
def home():  
    f = STATIC / "index.html"
    if not f.exists(): f = next(iter(STATIC.glob("index*")), None)
    if f is None: return JSONResponse({"error": f"Dashboard file not found. Put index.html inside {STATIC}"}, status_code=404)
    return FileResponse(f, media_type="text/html")

app.mount("/media", StaticFiles(directory="data"), name="media")
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")

if __name__ == "__main__":                         
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n  Smart CCTV is starting. Open http://localhost:{port}\n")
    threading.Timer(2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="0.0.0.0", port=port)
