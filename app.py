import logging
import gensim
import requests
import os
import flask
from flask import Flask, flash, request,jsonify, json
import json
import joblib
import praw
from gensim import utils
from bs4 import BeautifulSoup
import gensim.parsing.preprocessing as gsp

filters = [gsp.strip_tags,
gsp.strip_punctuation,
gsp.strip_multiple_whitespaces,
gsp.strip_numeric,
gsp.remove_stopwords,
gsp.strip_short,
gsp.stem_text
]

def clean(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

model = joblib.load(open('finalised_model.bin', 'rb'))


reddit = praw.Reddit(client_id = "5M0hvvospTusew",
                        client_secret = "-T_-9j9N_cyr1s5otX1m2_j-o69qHQ",
                        user_agent = "Reddit Flair Detection",
                        username = "prankh2403",
                        password = "fiatpunto2010")


def prediction(url):
    submission = reddit.submission(url = url)
    data = {}
    data["title"] = str(submission.title)
    data["url"] = str(submission.url)
    
    soup = BeautifulSoup(requests.get(data["url"]).content)
    string = ""           
    para_list = soup.find_all('p')
    for k in para_list:
        string += k.getText()
        
    data["url_page_text"] = string

    data['title'] = clean(str(data['title']))
    data['url_page_text'] = clean(str(data['url_page_text']))
    
    combined_features = data["title"] + data["url"] + data["url_page_text"]

    return model.predict([combined_features])

app = Flask(__name__,template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        text = flask.request.form['url']

        flair = str(prediction(str(text)))
        
        return flask.render_template('main.html', original_input={'url':str(text)}, result=flair[2:-2])
    
if __name__ == '__main__':
    app.run()