from flask import Flask, render_template, request
from search import retrieve
from time import time

app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Yandex',
        results=documents
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=88)
