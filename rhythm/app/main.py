import rhythm
from flask import Flask
from flask import request

app = Flask(__name__)

main_page = '''
<form method="get">
输入上句词语
<input type="text" name="line">
<br />
查看结果数量
<input type="text" name="num">
<br />
<input type="submit">
</form>
'''

answer_page = '''
<h1>%s</h1>
<br />
<h2>2-gram max probability satisfying rhythm</h2>
<br />
%s
<h2>nearest word vector satisfying rhythm</h2>
<br/>
%s
<br/>
'''

@app.route('/')
def hello_world():
    line = request.args.get('line')
    print(line)
    try:
        num = int(request.args.get('num'))
    except:
        num = None
    if line == None: return main_page
    if num == None:
        return answer_page % (line, str(rhythm.predict_phrase_2gram(line)), str(rhythm.predict_phrase_embedding(line)))
    else:
        return answer_page % (line, str(rhythm.predict_phrase_2gram(line, num)), str(rhythm.predict_phrase_embedding(line, num)))

if __name__ == '__main__':
  app.run(host='0.0.0.0')
