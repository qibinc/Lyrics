from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import rhythm
import seq2seq

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/word')
def generate_word():
    line = request.args.get('line').strip()
    if line == '': return jsonify([])
    ans = rhythm.predict_phrase_embedding(line, 100)
    ans = list(ans.values())[0]
    if ans == []:
        ans = rhythm.predict_phrase_2gram(line, 100)
    print(line, ans)
    return jsonify(ans)

@app.route('/seq2seq')
def generate_sentence():
    line = request.args.get('line').strip()
    if line == '': return jsonify([])
    ans = seq2seq.predict(line)
    print(line, ans)
    return jsonify(ans)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8001)
