from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess, pandas as pd, os, json, numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/run", methods=["POST"])
def run_analysis():
    try:
        username = request.json.get("username")
        if not username:
            return jsonify({"error": "Username required"}), 400

        # Pass username to backend
        os.environ["TARGET"] = username

        # Run both backend scripts
        subprocess.run(["python", "commentsextr.py"], check=True)
        subprocess.run(["python", "pipeline.py"], check=True)

        # Ensure all files exist
        required_files = [
            "top_hashtags.csv",
            "caption_type_performance.csv",
            "per_post_sentiments.csv"
        ]
        for f in required_files:
            if not os.path.exists(f):
                return jsonify({"error": f"Output file missing: {f}"}), 500

        # Helper: clean DataFrame and convert safely to JSON
        def clean_df(path):
            df = pd.read_csv(path).replace({np.nan: None})
            df = df.astype(object)
            return df.to_dict(orient="records")

        tags = clean_df("top_hashtags.csv")[:10]
        cap = clean_df("caption_type_performance.csv")
        posts = clean_df("per_post_sentiments.csv")

        response = {
            "username": username,
            "top_hashtags": tags,
            "caption_performance": cap,
            "per_post_sentiments": posts,
            "counts": {
                "hashtags": len(tags),
                "caption_types": len(cap),
                "posts": len(posts)
            }
        }

        # Ensure everything is JSON serializable
        json_ready = json.loads(json.dumps(response, default=str))
        return jsonify(json_ready)

    except subprocess.CalledProcessError as e:
        print("🔥 Script failed:", e)
        return jsonify({"error": f"Script failed: {e}"}), 500

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
