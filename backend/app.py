import os
import sys
import json
import time
import pickle
import base64
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from inference.model_loader import (
    run_pipeline_on_strokes,
    get_loaded_checkpoint_info,
    load_vocab_assets,
    reload_model_weights
)
from inference.solver import create_solution_output
from inference.latex import validate_math_expression
from inference.image_to_strokes import extract_strokes_from_image

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "processed_crohme.pkl")

app = Flask(__name__, static_folder=FRONTEND_DIR)

# Cached samples from validation set for quick UI testing
CACHED_SAMPLES = None

def get_demo_samples():
    global CACHED_SAMPLES
    if CACHED_SAMPLES is not None:
        return CACHED_SAMPLES

    samples = []
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "rb") as f:
                data, _ = pickle.load(f)

            # Target equations/expressions of interest
            targets = [
                ("x^2 + 5*x + 6 = 0", ["x", "^", "2", "+", "5", "x"]),
                ("2x + 5 = 15", ["2", "x", "+", "5", "="]),
                ("x + 5 = 12", ["x", "+", "5"]),
                ("\\frac{x}{2} + 3 = 7", ["\\frac", "x"]),
                ("\\phi(x)", ["\\phi"]),
                ("x^2 - 9 = 0", ["x", "^", "2", "-", "9"])
            ]

            found_keys = set()
            for idx, item in enumerate(data[:1000]):
                lbl = item.get("label", "").replace("$", "").strip()
                for label_name, search_terms in targets:
                    if label_name in found_keys:
                        continue
                    if any(t in lbl for t in search_terms):
                        samples.append({
                            "id": f"sample_{len(samples) + 1}",
                            "title": label_name,
                            "dataset_label": lbl,
                            "strokes": item["strokes"],
                            "stroke_count": len(item["strokes"]),
                            "point_count": sum(len(s) for s in item["strokes"])
                        })
                        found_keys.add(label_name)
                        break
                if len(samples) >= 6:
                    break
        except Exception as e:
            print(f"[MathInk.ai Samples] Notice: Could not extract sample strokes: {e}")

    # Fallback synthetic strokes if file unavailable
    if not samples:
        samples = [
            {
                "id": "sample_demo_1",
                "title": "2x + 5 = 15",
                "dataset_label": "2x + 5 = 15",
                "strokes": [
                    [[100, 100], [120, 90], [110, 120], [130, 120]],
                    [[150, 95], [170, 125]],
                    [[170, 95], [150, 125]],
                    [[190, 110], [210, 110]],
                    [[200, 100], [200, 120]],
                    [[225, 95], [240, 95], [225, 110], [240, 125]],
                    [[255, 105], [275, 105]],
                    [[255, 115], [275, 115]],
                    [[290, 95], [290, 125]],
                    [[305, 95], [320, 95], [305, 110], [320, 125]]
                ],
                "stroke_count": 10,
                "point_count": 25
            }
        ]

    CACHED_SAMPLES = samples
    return CACHED_SAMPLES

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/api/status", methods=["GET"])
def api_status():
    ckpt_info = get_loaded_checkpoint_info()
    vocab, t2i, _ = load_vocab_assets()

    return jsonify({
        "status": "online",
        "system": "MathInk.ai Smart Board Engine",
        "vocabulary_size": len(vocab),
        "checkpoint": ckpt_info["filename"],
        "checkpoint_loaded": ckpt_info["has_weights"],
        "checkpoint_path": ckpt_info["weights_path"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/reload_model", methods=["POST"])
def api_reload_model():
    success = reload_model_weights()
    ckpt_info = get_loaded_checkpoint_info()
    return jsonify({
        "success": success,
        "checkpoint": ckpt_info["filename"],
        "has_weights": ckpt_info["has_weights"]
    })

@app.route("/api/recognize_and_solve", methods=["POST"])
def api_recognize_and_solve():
    data = request.get_json(force=True, silent=True)
    if not data or "strokes" not in data:
        return jsonify({
            "success": False,
            "error": "No stroke data received."
        }), 400

    strokes = data.get("strokes", [])
    result = run_pipeline_on_strokes(strokes)
    return jsonify(result)

@app.route("/api/solve_latex", methods=["POST"])
def api_solve_latex():
    data = request.get_json(force=True, silent=True)
    if not data or "latex" not in data:
        return jsonify({
            "success": False,
            "error": "No LaTeX string provided."
        }), 400

    latex = data.get("latex", "").strip()
    if not latex:
        return jsonify({
            "success": False,
            "error": "LaTeX string cannot be empty."
        }), 400

    t0 = time.time()
    solution_payload = create_solution_output(latex)
    solve_time_ms = round((time.time() - t0) * 1000, 2)

    return jsonify({
        "success": True,
        "latex": latex,
        "expression_type": solution_payload.get("expression_type", "unknown"),
        "status": solution_payload.get("status", "unknown"),
        "solution": solution_payload.get("final_answer", ""),
        "steps": solution_payload.get("steps", []),
        "solve_time_ms": solve_time_ms
    })

@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    """
    Accepts an uploaded handwriting image (PNG/JPG/WEBP) or base64 data,
    extracts handwriting stroke sequences, and solves it.
    """
    image_bytes = None

    if "image" in request.files:
        image_bytes = request.files["image"].read()
    else:
        data = request.get_json(force=True, silent=True)
        if data and "image_base64" in data:
            raw_b64 = data["image_base64"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw_b64)
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64 encoding: {e}"}), 400

    if not image_bytes:
        return jsonify({"success": False, "error": "No image file or image data provided."}), 400

    try:
        strokes = extract_strokes_from_image(image_bytes, target_width=800, target_height=400)
    except Exception as e:
        return jsonify({"success": False, "error": f"Error processing image: {e}"}), 500

    if not strokes:
        return jsonify({"success": False, "error": "No handwriting or mathematical strokes detected in the image."}), 400

    result = run_pipeline_on_strokes(strokes)
    result["extracted_strokes"] = strokes
    return jsonify(result)

@app.route("/api/samples", methods=["GET"])
def api_samples():
    samples = get_demo_samples()
    # Return lightweight metadata + strokes
    return jsonify({
        "success": True,
        "count": len(samples),
        "samples": samples
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[MathInk.ai Server] Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
