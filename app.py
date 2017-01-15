from flask import Flask
from flask import render_template
from flask import request
from run import ai_system

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run(resume=None):
	answer = ai_system(request.form)
	print(answer)
	return request.form['resume']

if __name__ == "__main__":
    app.run()