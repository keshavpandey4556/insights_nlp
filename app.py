from flask import Flask, request, render_template
import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Initialize NLP components
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
vectorizer = CountVectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    
    # Tokenization
    tokens = word_tokenize(query_text)
    
    
    # Remove Stop Words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # LDA (Latent Dirichlet Allocation)
    data_vectorized = vectorizer.fit_transform(lemmatized_tokens)
    lda_model = LatentDirichletAllocation(n_components=1, random_state=0)
    lda_model.fit(data_vectorized)
    
    # NMF (Non-negative Matrix Factorization)
    nmf_model = NMF(n_components=1, random_state=0)
    nmf_model.fit(data_vectorized)
    
    # Named Entity Recognition (NER)
    doc = nlp(query_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Word Embeddings (Using spaCy)
    word_vectors = [nlp(word).vector for word in lemmatized_tokens]
    
    # Text Classification
    X = vectorizer.fit_transform(lemmatized_tokens)
    y = ["relevant" for _ in lemmatized_tokens]
    clf = MultinomialNB()
    clf.fit(X, y)
    
    # Search in Dataset
    df = pd.read_excel("data.xlsx", engine='openpyxl')  # Update the file path if needed
    
    relevant_columns = [col for col in df.columns if any(word.lower() in col.lower() for word in lemmatized_tokens)]
    
    relevant_data = pd.DataFrame()
    for token in lemmatized_tokens:
        relevant_data = pd.concat([relevant_data, df[df['Company name'].str.contains(token, case=False, na=False)]], ignore_index=True)
    
    relevant_data = relevant_data.drop_duplicates()
    
    # Display the Answer
    if relevant_data.empty:
        answer = "Sorry, I couldn't find any relevant information."
    else:
        result = relevant_data[relevant_columns].to_html(index=False)
        company_names = relevant_data['Company name'].unique()
        answer = f"<h3>Companies: {', '.join(company_names)}</h3>{result}"
    
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
