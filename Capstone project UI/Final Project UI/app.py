from __future__ import annotations

import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, UPLOAD_FOLDER
from phase1_inference import predict_phase1
from phase2_inference import predict_phase2


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_two_phase_pipeline(video_path: str) -> dict[str, object]:
    phase1_result = predict_phase1(video_path)

    if phase1_result["prediction_key"] == "camera":
        phase2_result = predict_phase2(video_path)
        final_label = phase2_result["prediction_label"]
        final_confidence = phase2_result["confidence"]
        pipeline_note = "Phase 1 predicted camera-captured content, so Phase 2 was executed."
    else:
        phase2_result = None
        final_label = "AI Generated"
        final_confidence = phase1_result["confidence"]
        pipeline_note = "Phase 1 predicted AI-generated / non-camera content, so Phase 2 was skipped."

    return {
        "final_label": final_label,
        "final_confidence": final_confidence,
        "pipeline_note": pipeline_note,
        "phase1": phase1_result,
        "phase2": phase2_result,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "video" not in request.files:
            error = "No video file part found in the request."
            return render_template("index.html", result=result, error=error)

        file = request.files["video"]
        if file.filename == "":
            error = "No video selected."
            return render_template("index.html", result=result, error=error)

        if not allowed_file(file.filename):
            error = "Unsupported file type. Upload mp4, avi, mov, mkv, or webm."
            return render_template("index.html", result=result, error=error)

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = UPLOAD_FOLDER / unique_name
        file.save(save_path)

        try:
            result = run_two_phase_pipeline(str(save_path))
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
