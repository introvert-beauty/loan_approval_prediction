from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


application=Flask(__name__)
app=application

# import the pickles of model,scaler
logistic_model=pickle.load(open("models/model.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def home():
    pass:
    


if __name__ == "__main__":
    app.run(debug=True)
