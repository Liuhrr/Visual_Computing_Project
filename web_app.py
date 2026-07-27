"""Flask backend serving the web-based Bonus Level frontend.

Streams processed camera frames as MJPEG and game state as SSE.
All game logic is delegated to WebGameSession, which reuses the existing
pose detection, scoring, and wall/freeze modules.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from web_game import WebGameSession

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web_bonus"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "SeeMeDoMyDance_saltyfrenchfry_0_9s_0.5x.mp4"
ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(__name__, static_folder=str(WEB_DIR / "static"), template_folder=str(WEB_DIR))
session = WebGameSession()


def _load_default_reference() -> None:
    if DEFAULT_VIDEO.is_file():
        try:
            session.load_reference(DEFAULT_VIDEO)
        except Exception as exc:
            print(f"Failed to auto-load default reference: {exc}")


# Load the default reference in the background so the landing page can offer
# a quick "Start Game" without waiting for a user upload.
threading.Thread(target=_load_default_reference, daemon=True).start()


# API routes must be registered BEFORE the catch-all static route.

@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "Empty filename"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO:
        return jsonify({"ok": False, "error": f"Unsupported format: {ext}"}), 400

    upload_path = PROJECT_ROOT / "data" / f"uploaded{ext}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    def analyze():
        try:
            session.load_reference(upload_path)
        except Exception as exc:
            session.state.model_error = f"Reference analysis failed: {exc}"

    threading.Thread(target=analyze, daemon=True).start()

    return jsonify({"ok": True, "message": "Analysis started"})


@app.route("/api/ready")
def ready():
    return jsonify({
        "ready": session.reference is not None,
        "duration": round(session.reference.duration, 2) if session.reference else 0.0,
        "frame_count": session.reference.frame_count if session.reference else 0,
        "error": session.state.model_error,
    })


@app.route("/api/start", methods=["POST"])
def start():
    if session.reference is None:
        return jsonify({"ok": False, "error": "Reference video not ready"}), 400
    ok = session.start()
    return jsonify({"ok": ok, "error": session.state.camera_error})


@app.route("/api/stop", methods=["POST"])
def stop():
    session.stop()
    return jsonify({"ok": True})


@app.route("/api/state")
def state_stream():
    def event_stream():
        last_json = ""
        while True:
            snap = session.get_state_snapshot()
            payload = {
                "running": snap.running,
                "finished": snap.finished,
                "elapsed": round(snap.elapsed, 2),
                "duration": round(snap.duration, 2),
                "dance_score": round(snap.dance_score, 1),
                "bonus_score": snap.bonus_score,
                "total_score": snap.total_score,
                "combo": snap.combo,
                "grade": snap.grade,
                "feedback": snap.feedback,
                "feedback_color": snap.feedback_color,
                "wall_state": snap.wall_state,
                "wall_match": round(snap.wall_match, 2),
                "freeze_active": snap.freeze_active,
                "freeze_message": snap.freeze_message,
                "camera_error": snap.camera_error,
                "model_error": snap.model_error,
                "pose_visible": snap.pose_visible,
                "fps": round(snap.fps, 1),
            }
            text = json.dumps(payload)
            if text != last_json:
                last_json = text
                yield f"data: {text}\n\n"
            if snap.finished:
                break
            time.sleep(0.05)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/video")
def video_stream():
    def generate():
        while True:
            jpeg = None
            with session._lock:
                jpeg = session.state.latest_frame_jpeg
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.02)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/final")
def final_state():
    snap = session.get_state_snapshot()
    return jsonify({
        "dance_score": round(snap.dance_score, 1),
        "bonus_score": snap.bonus_score,
        "total_score": snap.total_score,
        "grade": snap.grade,
        "feedback_counts": {},
    })


# Static files (must be last because it is a catch-all).

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
