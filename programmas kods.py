import cv2
import serial
import time
import mediapipe as mp
import threading
from flask import Flask, Response
from flask_socketio import SocketIO

# --- 0. Konfigurācija ----------------------------------------------
COM_PORT = 'COM4'
BAUD_RATE = 9600
DEAD_ZONE_WIDTH = 40

MIN_TRACKING_SPEED = 800
MAX_TRACKING_SPEED = 100000

# --- Koplietotais stāvoklis starp galveno lopu un Flask ------------
current_frame = None
frame_lock    = threading.Lock()
mode          = "manual" # "auto" — MI vada | "manual" — telefons vada
# -------------------------------------------------------------------

# --- 1. Serial savienojums -----------------------------------------
ser = None
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {COM_PORT}")
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error: Could not open serial port {COM_PORT}.")

def send_speed(speed_value):
    if ser and ser.isOpen():
        ser.write(f"{speed_value}\n".encode())

# --- 2. Kameras inicializācija -------------------------------------
cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Main camera failed, trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame")

frame_h, frame_w  = frame.shape[:2]
frame_center_x    = frame_w // 2
dead_zone_left    = frame_center_x - (DEAD_ZONE_WIDTH // 2)
dead_zone_right   = frame_center_x + (DEAD_ZONE_WIDTH // 2)
last_sent_speed   = 0

# --- 3. MediaPipe tracker ------------------------------------------
class MediaPipeFaceTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.last_box = None

    def update(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)

        if not results.detections:
            return False, None

        h, w, _ = frame.shape
        best_box = None
        min_dist = float('inf')

        if self.last_box:
            prev_cx = self.last_box[0] + self.last_box[2] / 2
            prev_cy = self.last_box[1] + self.last_box[3] / 2
        else:
            prev_cx, prev_cy = w / 2, h / 2

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x  = int(bboxC.xmin  * w)
            y  = int(bboxC.ymin  * h)
            bw = int(bboxC.width  * w)
            bh = int(bboxC.height * h)
            cx = x + bw / 2
            cy = y + bh / 2
            dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_box = (x, y, bw, bh)

        if best_box:
            self.last_box = best_box
            return True, best_box

        return False, None

def map_value(value, from_low, from_high, to_low, to_high):
    value = max(from_low, min(from_high, value))
    from_span    = from_high - from_low
    to_span      = to_high   - to_low
    value_scaled = float(value - from_low) / float(from_span)
    return int(to_low + (value_scaled * to_span))

# Tracker startē uzreiz — MediaPipe pats atrod seju, ROI nav vajadzīgs
tracker = MediaPipeFaceTracker()

# ===================================================================
# FLASK SERVERIS
# ===================================================================

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML_PAGE = """<!DOCTYPE html>
<html lang="lv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Kameras vadība</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; touch-action: manipulation; }
  body {
    background: #111; color: #eee;
    font-family: system-ui, sans-serif;
    display: flex; flex-direction: column; height: 100dvh; overflow: hidden;
  }
  #video-wrap { position: relative; flex: 1; background: #000; overflow: hidden; }
  #stream { width: 100%; height: 100%; object-fit: contain; }
  #status-badge {
    position: absolute; top: 10px; left: 10px;
    padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 500;
    background: rgba(0,0,0,0.55); border: 1.5px solid currentColor;
    transition: color .3s, border-color .3s;
  }
  #status-badge.auto   { color: #4ade80; }
  #status-badge.manual { color: #f87171; }
  #latency { position: absolute; top: 10px; right: 10px; font-size: 11px; color: rgba(255,255,255,.45); }
  #controls { background: #1a1a1a; padding: 14px 16px 20px; display: flex; flex-direction: column; gap: 12px; }
  #mode-btn {
    width: 100%; padding: 13px; border-radius: 10px; border: none;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: background .25s, color .25s;
  }
  #mode-btn.auto   { background: #166534; color: #bbf7d0; }
  #mode-btn.manual { background: #7f1d1d; color: #fecaca; }
  #arrow-row { display: flex; gap: 10px; }
  .arrow-btn {
    flex: 1; height: 64px; border-radius: 10px; border: none;
    background: #2a2a2a; color: #e5e5e5; font-size: 28px; cursor: pointer;
    user-select: none; -webkit-user-select: none;
    transition: background .15s, transform .1s;
    display: flex; align-items: center; justify-content: center;
  }
  .arrow-btn:active { background: #3a3a3a; transform: scale(0.96); }
  .arrow-btn:disabled { opacity: 0.28; }
  #speed-row { display: flex; align-items: center; gap: 10px; font-size: 13px; color: #888; }
  #speed-slider { flex: 1; accent-color: #60a5fa; height: 6px; }
  #speed-val { min-width: 56px; text-align: right; color: #ccc; font-size: 13px; }
</style>
</head>
<body>
<div id="video-wrap">
  <img id="stream" src="/video" alt="Video strīms">
  <div id="status-badge" class="auto">&#9679; AUTO</div>
  <div id="latency"></div>
</div>
<div id="controls">
  <button id="mode-btn" class="auto" onclick="toggleMode()">
    &#129302; AUTO &mdash; pieskarties, lai pārslēgtu
  </button>
  <div id="arrow-row">
    <button class="arrow-btn" id="btn-left"
            onpointerdown="startMove('L')" onpointerup="stopMove()"
            onpointerleave="stopMove()" disabled>&#8592;</button>
    <button class="arrow-btn" id="btn-stop"
            onclick="stopMove()" disabled>&#9632;</button>
    <button class="arrow-btn" id="btn-right"
            onpointerdown="startMove('R')" onpointerup="stopMove()"
            onpointerleave="stopMove()" disabled>&#8594;</button>
  </div>
  <div id="speed-row">
    <span>Ātrums</span>
    <input type="range" id="speed-slider" min="1000" max="30000" value="10000" step="1000"
           oninput="document.getElementById('speed-val').textContent=this.value">
    <span id="speed-val">20000</span>
  </div>
</div>
<script>
  const socket = io();
  let currentMode = "auto";
  let moveInterval = null;

  socket.on("connect",    () => document.getElementById("latency").textContent = "savienots");
  socket.on("disconnect", () => document.getElementById("latency").textContent = "atvienots");
  socket.on("status",     d  => applyMode(d.mode));

  setInterval(() => {
    const t0 = Date.now();
    socket.emit("ping_ts", {}, () => {
      document.getElementById("latency").textContent = (Date.now()-t0)+"ms";
    });
  }, 2000);

  function toggleMode() {
    socket.emit("set_mode", { value: currentMode === "auto" ? "manual" : "auto" });
  }

  function applyMode(m) {
    currentMode = m;
    const btn   = document.getElementById("mode-btn");
    const badge = document.getElementById("status-badge");
    const btns  = document.querySelectorAll(".arrow-btn");
    if (m === "auto") {
      btn.className = "auto";
      btn.innerHTML = "&#129302; AUTO &mdash; pieskarties, lai pārslēgtu";
      badge.className = "auto"; badge.textContent = "\u25CF AUTO";
      btns.forEach(b => b.disabled = true);
    } else {
      btn.className = "manual";
      btn.innerHTML = "&#128308; MANUAL &mdash; pieskarties, lai pārslēgtu";
      badge.className = "manual"; badge.textContent = "\u25CF MANUAL";
      btns.forEach(b => b.disabled = false);
    }
  }

  function getSpeed() { return parseInt(document.getElementById("speed-slider").value); }

  function startMove(dir) {
    if (currentMode !== "manual") return;
    sendMove(dir);
    moveInterval = setInterval(() => sendMove(dir), 120);
  }
  function stopMove() {
    clearInterval(moveInterval); moveInterval = null;
    socket.emit("move", { dir: "S", speed: 0 });
  }
  function sendMove(dir) { socket.emit("move", { dir, speed: getSpeed() }); }

  document.addEventListener("pointercancel", stopMove);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/video")
def video():
    def generate():
        while True:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                f = current_frame.copy()
            _, jpeg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg.tobytes() + b"\r\n")
            time.sleep(1 / 30)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@socketio.on("connect")
def on_connect():
    socketio.emit("status", {"mode": mode})

@socketio.on("set_mode")
def on_set_mode(data):
    global mode
    mode = data.get("value", "auto")
    if mode == "manual":
        send_speed(0)
    socketio.emit("status", {"mode": mode})

@socketio.on("move")
def on_move(data):
    if mode != "manual":
        return
    direction = data.get("dir", "S")
    speed     = int(data.get("speed", 0))
    if direction == "L":
        send_speed(-speed)
    elif direction == "R":
        send_speed(speed)
    else:
        send_speed(0)

@socketio.on("ping_ts")
def on_ping(data):
    pass

def run_flask():
    print("[Flask] Serveris: http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# ===================================================================
# GALVENĀ LOPA
# ===================================================================

print("[i] 'r' - Pauzēt / atsākt sekošanu")
print("[i] 'q' - Iziet")

paused = False

while True:
    ok, frame = cap.read()
    if not ok:
        break

    final_speed = 0

    if mode == "auto" and not paused:
        ok_track, roi = tracker.update(frame)
        if ok_track:
            x, y, w, h = map(int, roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

            target_center_x = x + (w // 2)
            error = target_center_x - frame_center_x

            if abs(error) > (DEAD_ZONE_WIDTH // 2):
                abs_error  = abs(error)
                final_speed = map_value(abs_error,
                                        (DEAD_ZONE_WIDTH // 2), frame_center_x,
                                        MIN_TRACKING_SPEED, MAX_TRACKING_SPEED)
                if error < 0:
                    final_speed = -final_speed
        else:
            cv2.putText(frame, "LOST!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if final_speed != last_sent_speed:
            send_speed(final_speed)
            last_sent_speed = final_speed

    # Statusa teksts uz PC loga
    if paused:
        cv2.putText(frame, "PAUZE  —  'r' lai atsāktu", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    elif mode == "manual":
        cv2.putText(frame, "MANUAL", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.line(frame, (dead_zone_left,  0), (dead_zone_left,  frame_h), (0, 255, 255), 1)
    cv2.line(frame, (dead_zone_right, 0), (dead_zone_right, frame_h), (0, 255, 255), 1)

    with frame_lock:
        current_frame = frame.copy()

    cv2.imshow("Kameras vadība (MediaPipe)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        paused = not paused
        if paused:
            send_speed(0)
            last_sent_speed = 0
            print("[Info] Pauze.")
        else:
            print("[Info] Sekošana atsākta.")

    elif key == ord('q'):
        break

# --- Tīrīšana -----------------------------------------------------
send_speed(0)
if ser and ser.isOpen():
    ser.close()
cap.release()
cv2.destroyAllWindows()