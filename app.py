from flask import Flask, render_template, request, jsonify
from research import run_research
import asyncio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research():
    query = request.form['query']
    pdf_path, paper_titles = asyncio.run(run_research(query))
    if pdf_path:
        return jsonify({'pdf_path': pdf_path, 'paper_titles': paper_titles})
    else:
        return jsonify({'error': 'Could not generate the report.'})

if __name__ == '__main__':
    app.run(debug=True)
