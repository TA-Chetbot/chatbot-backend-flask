from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_model import preprocess_question, generate_text_sampling_top_p_nucleus_22

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def read_root():
    response = {"message": "hello world", "status": "ok"}
    return jsonify(response)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    question = data.get('question')
    if question:
        question = preprocess_question(question)
        answer = generate_text_sampling_top_p_nucleus_22(question)
        return jsonify({"preprocessed_question": question, "answer": answer})
    else:
        return jsonify({"error": "No question provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)