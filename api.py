# Python Libs
from flask import Flask
from flask_restful import Api, request
import json
import sklearn
import pickle
from html.parser import HTMLParser
import re
import unicodedata
import html
import nltk

# Create Flask APP
app = Flask("Sentiment Analyses")
api = Api(app)

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_citation(text):
    return ' '.join(re.sub("(@[A-Za-z0-9_]+)"," ",text).split())

def remove_https(text):
    return re.sub(r'http\S+', '', text)

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_diacritics(text):
    text = unicodedata.normalize('NFKD',str.encode(text).decode('utf-8')).encode('ascii', 'ignore')
    return text

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
        self.convert_charrefs=[]
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_html(text):
    html_stripper = MLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

def normalize_tweet(tweet, only_text_chars=False):

     tweet = remove_citation(tweet)
     tweet = remove_https(tweet)
     tweet = remove_diacritics(tweet)
     tweet = html.parser.unescape(tweet.decode())
     tweet = strip_html(tweet)
                
     if only_text_chars:
         tweet = keep_text_characters(tweet)
         
     return tweet 

# This route will receive something via POST method
@app.route("/classify", methods=["POST"])
def classify_tweet():

    # JSON for the return

    # Your text variable sent via POST
    # myinput = [request.values["text"]]
    tweet = request.data.decode('utf-8')

    norm_tweet = normalize_tweet(tweet, only_text_chars=False)
    print('norm_tweet:',norm_tweet)

    # Getting back the vectorizer:
    vectorizer = pickle.load( open( "vectorizer.pickle", "rb" ) )
    tweet_features = vectorizer.transform([norm_tweet])
    
    # Getting model
    clf = pickle.load(open("model_sdg_svm.pickle", "rb"))
    sentiment=int(clf.predict(tweet_features)[0])

    if sentiment == 0:
        sentiment = "Negativo"
    if sentiment == 1:
        sentiment = "Positivo"
    if sentiment == 2:
        sentiment = "Neutro"

    output = {'sentiment':sentiment}

    # JSON output
    return json.dumps(output)

# Your Flask API host adress
app.run(host="0.0.0.0", port=5000, debug=True)
