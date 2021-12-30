from flask import Flask, render_template, request
from Database import Database
from time import time

database = Database()

database.load_database()
database.load_tfidf_body_vectorizer()
database.load_tfidf_title_vectorizer()
database.load_title_inverted_index()
database.load_body_inverted_index()

app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')

    if query is None:
        query = ''

    documents = database.find_n_best_docs(query=query, n=20)

    return render_template(
        'index.html',
        time="%.2f" % (time() - start_time),
        query=query,
        search_engine_name='Stack Overflow',
        results=documents
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=88)
