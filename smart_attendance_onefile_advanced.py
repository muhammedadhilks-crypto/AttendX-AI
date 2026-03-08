import os, time, csv, socket, threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, send_file

# =========================
# CONFIG
# =========================
DEMO_SECONDS = 100
THRESHOLD_PERCENT = 60

SAMPLES_PER_STUDENT = 30
LBPH_ACCEPT_THRESHOLD = 90
BIND_CONF_THRESHOLD = 110

REFRESH_DETECT_EVERY = 12
MIN_FACE_SIZE = (80, 80)

DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CSV_PATH = os.path.join(DATA_DIR, "students.csv")
MODEL_PATH = os.path.join(DATA_DIR, "lbph_model.yml")
REPORTS_DIR = "reports"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# UI (single source of truth)
# =========================
HEADER = (0, 0, 0)            # BGR
ACCENT = (255, 149, 0)        # premium orange (BGR)
TEXT = (235, 235, 235)
SUBTEXT = (135, 135, 135)
GREEN_UI = (90, 200, 120)
RED_UI = (60, 60, 220)        # red-ish BGR

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL


def txt(img, s, x, y, scale=0.8, color=TEXT, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def pill(img, x, y, w, h, bg, text, tcolor=TEXT):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2 - 2
    txt(img, text, tx, ty, 0.55, tcolor, 1)


def draw_header(frame):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), HEADER, -1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 4), ACCENT, -1)

    title = "AttendX"
    txt(frame, title, 20, 38, 1.0, TEXT, 1)

    # AI badge auto-positioned next to title
    (tw, th), _ = cv2.getTextSize(title, FONT, 1.0, 1)
    badge_x = 20 + tw + 10
    badge_y = 18
    badge_w = 34
    badge_h = 18

    cv2.rectangle(frame, (badge_x, badge_y),
                  (badge_x + badge_w, badge_y + badge_h),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (badge_x, badge_y),
                  (badge_x + badge_w, badge_y + badge_h),
                  ACCENT, 1)
    txt(frame, "AI", badge_x + 8, badge_y + 14, 0.50, (255, 255, 255), 1)

    txt(frame, "R:Register   T:Train   A:Attendance   ESC:Exit", 20, 70, 0.60, SUBTEXT, 1)


# =========================
# ArUco
# =========================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# =========================
# Face detector
# =========================
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# =========================
# Students storage
# =========================
def load_students():
    students = {}
    if not os.path.exists(CSV_PATH):
        return students
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["id"])
            students[sid] = {"roll": row["roll_no"], "name": row["name"]}
    return students


def upsert_student(sid: int, roll_no: str, name: str):
    students = load_students()
    students[sid] = {"roll": roll_no, "name": name}
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "roll_no", "name"])
        for k in sorted(students.keys()):
            w.writerow([k, students[k]["roll"], students[k]["name"]])


STUDENTS = load_students()


def ensure_student_folder(sid: int):
    folder = os.path.join(DATASET_DIR, str(sid))
    os.makedirs(folder, exist_ok=True)
    return folder


def count_samples(sid: int) -> int:
    folder = os.path.join(DATASET_DIR, str(sid))
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def save_face_sample(gray, face_box, sid: int) -> int:
    x, y, w, h = face_box
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (200, 200))
    roi = cv2.equalizeHist(roi)
    folder = ensure_student_folder(sid)
    idx = count_samples(sid) + 1
    cv2.imwrite(os.path.join(folder, f"{idx:03d}.png"), roi)
    return idx


# =========================
# Model (LBPH)  (requires opencv-contrib)
# =========================
def train_lbph():
    images, labels = [], []
    for sid_name in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, sid_name)
        if not os.path.isdir(folder):
            continue
        try:
            sid = int(sid_name)
        except:
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".png"):
                img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(sid)

    if len(images) < 10:
        return None, "Not enough samples. Register more faces first."

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(images, np.array(labels, dtype=np.int32))
    rec.save(MODEL_PATH)
    return rec, f"Trained. Samples={len(images)}, Students={len(set(labels))}"


recognizer = None
if os.path.exists(MODEL_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)


def recognize(gray, face_box):
    if recognizer is None:
        return None, None
    x, y, w, h = face_box
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (200, 200))
    roi = cv2.equalizeHist(roi)
    label, conf = recognizer.predict(roi)
    return int(label), int(conf)


# =========================
# Tracking helpers
# =========================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a2x, b2x), min(a2y, b2y)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union


def center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


# =========================
# Faculty dashboard (Flask)
# =========================
app = Flask(__name__)
state_lock = threading.Lock()
state = {
    "mode": "IDLE",
    "note": "Keys: R register | T train | A attendance | ESC exit",
    "running": False,
    "remaining": DEMO_SECONDS,
    "last_report": None,
    "table": [],
    "live": {"aruco": None, "bind": None, "conf": None}
}

PAGE = """
<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AttendX Faculty</title>
<style>
:root{--bg:#f3f4f6;--card:#fff;--text:#111827;--muted:#6b7280;--primary:#2563eb;--ok:#16a34a;--bad:#dc2626;--border:#e5e7eb;}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--text);padding:12px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:14px;margin:0 0 12px 0;box-shadow:0 6px 18px rgba(0,0,0,.06);}
.big{font-size:16px;font-weight:800;color:var(--primary);margin-bottom:6px;}
.muted{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:8px;border-bottom:1px solid var(--border)}
.ok{color:var(--ok);font-weight:900}
.bad{color:var(--bad);font-weight:900}
</style></head>
<body>
<div class="card">
  <div class="big">AttendX Faculty Dashboard</div>
  <div class="muted">Rule: ≥ {{thr}}% ⇒ PRESENT | Session: {{sec}}s</div>
  <div class="muted">Mode: <span id="mode"></span> | Remaining: <span id="rem"></span>s</div>
  <div class="muted" id="note"></div>
</div>
<div class="card">
  <div class="big">Live</div>
  <div class="muted">Aruco: <span id="aru"></span> | Bound: <span id="bnd"></span> | conf: <span id="cf"></span></div>
  <div class="muted">Report: <a href="/report">Download CSV</a> (<span id="rep"></span>)</div>
</div>
<div class="card">
  <div class="big">Class Table</div>
  <table>
    <thead><tr><th>ID</th><th>Roll</th><th>Name</th><th>Sec</th><th>%</th><th>Status</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>
<script>
async function tick(){
  const s = await (await fetch('/api')).json();
  mode.innerText = s.mode;
  rem.innerText = s.remaining;
  note.innerText = s.note;
  aru.innerText = s.live.aruco ?? "-";
  bnd.innerText = s.live.bind ?? "-";
  cf.innerText  = s.live.conf ?? "-";
  rep.innerText = s.last_report ?? "none";

  rows.innerHTML = "";
  for(const r of s.table){
    const st = (r.status==="PRESENT") ? "<span class='ok'>PRESENT</span>" : "<span class='bad'>ABSENT</span>";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.id}</td><td>${r.roll}</td><td>${r.name}</td><td>${r.sec.toFixed(1)}</td><td>${r.pct.toFixed(1)}</td><td>${st}</td>`;
    rows.appendChild(tr);
  }
}
setInterval(tick, 500); tick();
</script>
</body></html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE, thr=THRESHOLD_PERCENT, sec=DEMO_SECONDS)


@app.get("/api")
def api():
    with state_lock:
        return jsonify(state)


@app.get("/report")
def report():
    with state_lock:
        path = state.get("_report_path")
        name = state.get("last_report") or "attendance.csv"
    if not path or not os.path.exists(path):
        return ("No report yet.", 404)
    return send_file(path, as_attachment=True, download_name=name)


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def run_dashboard():
    ip = local_ip()
    print(f"\nFACULTY PHONE LINK (same Wi-Fi/hotspot): http://{ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


threading.Thread(target=run_dashboard, daemon=True).start()

# =========================
# Report
# =========================
def make_table(presence_by_id):
    students = load_students()
    rows = []
    ids_ = sorted(set(list(students.keys()) + list(presence_by_id.keys())))
    for sid in ids_:
        info = students.get(sid, {"roll": str(sid), "name": f"ID_{sid}"})
        sec = float(presence_by_id.get(sid, 0.0))
        pct = (sec / DEMO_SECONDS) * 100.0
        status = "PRESENT" if pct >= THRESHOLD_PERCENT else "ABSENT"
        rows.append({"id": sid, "roll": info["roll"], "name": info["name"], "sec": sec, "pct": pct, "status": status})
    return rows


def write_report(presence_by_id):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"attendance_{ts}.csv"
    path = os.path.join(REPORTS_DIR, fname)
    rows = make_table(presence_by_id)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "roll", "name", "presence_seconds", "presence_percent", "result", "session_seconds"])
        for r in rows:
            w.writerow([r["id"], r["roll"], r["name"], f'{r["sec"]:.1f}', f'{r["pct"]:.1f}', r["status"], DEMO_SECONDS])
    with state_lock:
        state["last_report"] = fname
        state["_report_path"] = path
        state["table"] = rows
    return path


# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Close other camera apps and retry.")

trackers = {}
next_tid = 1
track_boxes = {}
bound_tid_to_id = {}
presence_by_id = {}

mode = "IDLE"
attendance_running = False
start_t = 0.0
last_t = 0.0
frame_i = 0

verified_until = 0.0  # show VERIFIED until this time (then hide)


def add_tracker(frame, box):
    global next_tid
    tr = cv2.TrackerCSRT_create()
    tr.init(frame, tuple(box))
    tid = next_tid
    trackers[tid] = tr
    next_tid += 1
    return tid


print("====================================================")
print("AttendX FIXED ONE-FILE APP")
print("Keys (OpenCV window focused): R register | T train | A attendance | ESC exit")
print("====================================================")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_i += 1

    # Header always
    draw_header(frame)

    # --- ArUco detect ---
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    aruco_id = None
    aruco_box = None
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        aruco_id = int(ids.flatten()[0])
        pts = corners[0].reshape(-1, 2)
        x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        x2, y2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
        aruco_box = (x1, y1, x2 - x1, y2 - y1)

    # --- Update trackers ---
    track_boxes.clear()
    dead = []
    for tid, tr in trackers.items():
        ok, bb = tr.update(frame)
        if not ok:
            dead.append(tid)
        else:
            x, y, w, h = map(int, bb)
            track_boxes[tid] = (x, y, w, h)

    for tid in dead:
        trackers.pop(tid, None)
        track_boxes.pop(tid, None)
        bound_tid_to_id.pop(tid, None)

    # --- Periodic face detect to add trackers ---
    if frame_i % REFRESH_DETECT_EVERY == 0:
        faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5, minSize=MIN_FACE_SIZE)
        for (x, y, w, h) in faces:
            cand = (x, y, w, h)
            matched = False
            for box in track_boxes.values():
                if iou(cand, box) > 0.3:
                    matched = True
                    break
            if not matched:
                add_tracker(frame, cand)

    # =========================
    # REGISTER MODE
    # =========================
    if mode == "REGISTER":
        with state_lock:
            state["mode"] = "REGISTER"
            state["live"]["aruco"] = aruco_id
            state["live"]["bind"] = None
            state["live"]["conf"] = None

        if aruco_id is None:
            with state_lock:
                state["note"] = "Show ArUco marker to register."
            txt(frame, "REGISTER: SHOW ARUCO", 20, 130, 0.8, RED_UI, 1)
        else:
            if aruco_id not in STUDENTS:
                upsert_student(aruco_id, str(aruco_id), f"Student_{aruco_id}")
                STUDENTS = load_students()

            best_tid, best_d = None, 1e18
            if aruco_box is not None:
                acx, acy = center(aruco_box)
                for tid, box in track_boxes.items():
                    cx, cy = center(box)
                    d = (cx - acx) ** 2 + (cy - acy) ** 2
                    if d < best_d:
                        best_d = d
                        best_tid = tid

            if best_tid is None or best_tid not in track_boxes:
                with state_lock:
                    state["note"] = "Stand close so face is tracked near marker."
                txt(frame, "REGISTER: MOVE FACE NEAR MARKER", 20, 130, 0.75, RED_UI, 1)
            else:
                saved = count_samples(aruco_id)
                if saved < SAMPLES_PER_STUDENT:
                    idx = save_face_sample(gray, track_boxes[best_tid], aruco_id)
                    saved = idx
                    with state_lock:
                        state["note"] = f"Registering ID {aruco_id}: saved {saved}/{SAMPLES_PER_STUDENT}"
                    txt(frame, f"REGISTER ID {aruco_id}: {saved}/{SAMPLES_PER_STUDENT}", 20, 130, 0.85, GREEN_UI, 1)
                    time.sleep(0.05)
                else:
                    with state_lock:
                        state["note"] = f"Registered ID {aruco_id}. Press T to train or R to exit register."
                    txt(frame, f"REGISTER DONE: {aruco_id}", 20, 130, 0.85, GREEN_UI, 1)

    # =========================
    # ATTENDANCE MODE
    # =========================
    if attendance_running:
        dt = now - last_t
        last_t = now

        elapsed = now - start_t
        remaining = max(0.0, DEMO_SECONDS - elapsed)

        # Mini timer top-right (tiny)
        rem = int(remaining)
        w, h = 48, 22
        x = frame.shape[1] - w - 12
        y = 10
        pill(frame, x, y, w, h, ACCENT, str(rem))

        bound_now = None
        conf_now = None

        # Bind once (optional): if ArUco shown, bind nearest track to that ID
        if aruco_id is not None and aruco_box is not None:
            acx, acy = center(aruco_box)
            best_tid, best_d = None, 1e18
            for tid, box in track_boxes.items():
                cx, cy = center(box)
                d = (cx - acx) ** 2 + (cy - acy) ** 2
                if d < best_d:
                    best_d = d
                    best_tid = tid

            if best_tid is not None:
                label, conf = recognize(gray, track_boxes[best_tid])
                conf_now = conf
                if label == aruco_id and conf is not None and conf <= BIND_CONF_THRESHOLD:
                    bound_tid_to_id[best_tid] = aruco_id
                    bound_now = aruco_id
                    verified_until = time.time() + 2.0  # show VERIFIED briefly

        # TIME COUNTING
        # 1) If bound -> count only if face matches bound identity (strict)
        for tid, sid in list(bound_tid_to_id.items()):
            if tid not in track_boxes:
                continue
            label, conf = recognize(gray, track_boxes[tid])
            if label == sid and conf is not None and conf <= LBPH_ACCEPT_THRESHOLD:
                presence_by_id[sid] = presence_by_id.get(sid, 0.0) + dt

        # 2) If no bound yet -> fallback face-only (so you don't "need" ArUco forever)
        if len(bound_tid_to_id) == 0:
            for tid, box in track_boxes.items():
                label, conf = recognize(gray, box)
                if label is not None and conf is not None and conf <= LBPH_ACCEPT_THRESHOLD:
                    presence_by_id[label] = presence_by_id.get(label, 0.0) + dt

        # Dashboard update
        rows = make_table(presence_by_id)
        with state_lock:
            state["mode"] = "ATTENDANCE"
            state["running"] = True
            state["remaining"] = int(remaining)
            state["table"] = rows
            state["live"]["aruco"] = aruco_id
            state["live"]["bind"] = bound_now
            state["live"]["conf"] = conf_now
            state["note"] = "Bind once: show ArUco+face (1–2s). After bind: face-only time counts."

        # UI message:
        # show bind instruction only until first binding exists
        if len(bound_tid_to_id) == 0:
            txt(frame, "SHOW ARUCO NEAR FACE (ONCE) TO BIND", 20, 170, 0.70, RED_UI, 1)
        else:
            # show VERIFIED only briefly
            if time.time() < verified_until:
                txt(frame, "VERIFIED", 20, 170, 0.85, GREEN_UI, 1)

        if bound_now is not None:
            txt(frame, f"BOUND NOW: ID {bound_now}", 20, 205, 0.75, GREEN_UI, 1)

        # Finish
        if elapsed >= DEMO_SECONDS:
            attendance_running = False
            path = write_report(presence_by_id)
            with state_lock:
                state["running"] = False
                state["remaining"] = 0
                state["mode"] = "FINISHED"
                state["note"] = f"Finished. Report ready: {state['last_report']}."
            print("REPORT SAVED:", path)

    # --- draw tracking boxes ---
    for tid, box in track_boxes.items():
        x, y, w, h = box
        sid = bound_tid_to_id.get(tid)
        if sid is None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), ACCENT, 1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN_UI, 2)
            name = load_students().get(sid, {"name": str(sid)}).get("name", str(sid))
            txt(frame, f"{name} (ID {sid})", x, max(95, y - 8), 0.75, TEXT, 1)

    # Show window
    cv2.imshow("AttendX AI", frame)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

    # Keys (OpenCV window must be focused)
    if k in (ord('r'), ord('R')):
        mode = "REGISTER" if mode != "REGISTER" else "IDLE"
        with state_lock:
            state["mode"] = mode
            state["note"] = "REGISTER: show ArUco + keep face near marker." if mode == "REGISTER" else "IDLE"

    if k in (ord('t'), ord('T')):
        rec, msg = train_lbph()
        if rec is not None:
            recognizer = rec
        with state_lock:
            state["mode"] = "TRAINED" if recognizer is not None else "IDLE"
            state["note"] = msg
        print(msg)

    if k in (ord('a'), ord('A')):
        if recognizer is None:
            with state_lock:
                state["note"] = "Train first (T). Register faces (R) if needed."
            print("Train first (T).")
        else:
            attendance_running = True
            start_t = time.time()
            last_t = start_t
            presence_by_id = {}
            bound_tid_to_id = {}
            verified_until = 0.0
            with state_lock:
                state["mode"] = "ATTENDANCE"
                state["running"] = True
                state["remaining"] = DEMO_SECONDS
                state["note"] = "Attendance started."
            print("Attendance started.")

cap.release()
cv2.destroyAllWindows()
