from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

app=Flask(__name__)
Swagger(app)

model = pickle.load(open('SVM.pkl','rb'))
countVect = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('index.html', Reviews=Reviews, prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
    