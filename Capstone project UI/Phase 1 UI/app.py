import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from inference import predict_video

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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
            result = predict_video(str(save_path))
        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)


