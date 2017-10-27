import rhythm
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/word')
def generate_word():
    line = request.args.get('line')
    try:
        num = int(request.args.get('num'))
    except:
        num = None
    print(line, num)
    if num == None:
        return str(rhythm.predict_phrase_2gram(line)) + str(rhythm.predict_phrase_embedding(line))
    else:
        return str(rhythm.predict_phrase_2gram(line, num)) + str(rhythm.predict_phrase_embedding(line, num))

@app.route('/sentence')
def generate_sentence():
    line = request.args.get('line')
    try:
        num = int(request.args.get('num'))
    except:
        num = None
    if num == None:
        return (line, str(rhythm.predict_phrase_2gram(line)), str(rhythm.predict_phrase_embedding(line)))
    else:
        return (line, str(rhythm.predict_phrase_2gram(line, num)), str(rhythm.predict_phrase_embedding(line, num)))

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
