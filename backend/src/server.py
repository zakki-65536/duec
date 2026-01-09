# app.py
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# run_chat.py を編集せずにそのまま利用（import時に末尾のprintが走る点に注意）
import run_chat

app = Flask(__name__)
CORS(app)

def parse_history_param(current_history_str: str):
    """
    current_history は GET パラメータで JSON 文字列として受け取る想定。
    例: [["user","..."],["assistant","..."]]
    戻り値は [(role, text), ...]
    """
    if not current_history_str:
        return []
    data = json.loads(current_history_str)
    if not isinstance(data, list):
        raise ValueError("current_history must be a JSON array")
    history = []
    for i, item in enumerate(data):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError(f"current_history[{i}] must be [role, text] (both strings)")
        history.append((item[0], item[1]))
    return history

@app.get("/chat")
def chat():
    new_input = request.args.get("new_input", default=None, type=str)
    if not new_input:
        return jsonify(error="new_input is required"), 400

    current_history_str = request.args.get("history", default="[]", type=str)
    try:
        history = parse_history_param(current_history_str)
    except Exception as e:
        return jsonify(error=f"invalid current_history: {e}"), 400

    # run_chat.py の関数をそのまま呼ぶ
    response_text = run_chat.get_ai_response_one_shot(history, new_input)

    return Response(response_text, mimetype="text/plain; charset=utf-8")

@app.get("/healthz")
def healthz():
    return 'status="ok"'

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

if __name__ == "__main__":
    app.run(debug=True)

