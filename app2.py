from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your DataFrame
df = pd.read_csv('BooksDatasetClean.csv')  # Make sure to adjust the filename

# Preprocess the data
df.fillna('', inplace=True)
df['text'] = df['Title'] + ' ' + df['Authors'] + ' ' + df['Category'] + ' ' + df['Description']

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])



# Function to preprocess and vectorize user query
def preprocess_query(query):
    return vectorizer.transform([query])

# Function to search books
def search_books(query, top_n=20):
    query_vector = preprocess_query(query)
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    top_n_indices = np.argsort(cosine_similarities)[-top_n:]

    results = df.iloc[top_n_indices][['Title', 'Authors', 'Description', 'Category', 'Publisher', 'Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)']]

    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 100
    total = len(df)
    
    if request.method == 'POST':
        user_query = request.form['search']
        search_results = search_books(user_query)
        return render_template('results.html', results=search_results)
    else:
        books = df.iloc[(page - 1) * per_page: page * per_page]
        return render_template('index.html', books=books, page=page, per_page=per_page, total=total)

@app.route('/results', methods=['POST'])
def results():
    user_query = request.form['search']
    search_results = search_books(user_query)
    return render_template('results.html', results=search_results)

@app.context_processor
def utility_processor():
    return dict(max=max, min=min)

if __name__ == '__main__':
    app.run(debug=True)
