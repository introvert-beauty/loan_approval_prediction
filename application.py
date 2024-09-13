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



#  Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area
@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Gender=int(request.form.get("Gender"))
        Married=int(request.form.get("Married"))
        Dependents=int(request.form.get("Dependents"))
        Education=int(request.form.get("Education"))
        Self_Employed=int(request.form.get("Self_Employed"))
        ApplicantIncome=int(request.form.get("ApplicantIncome"))
        CoapplicantIncome=float(request.form.get("CoapplicantIncome"))
        LoanAmount=float(request.form.get("LoanAmount"))
        Loan_Amount_Term=float(request.form.get("Loan_Amount_Term"))
        Credit_History=float(request.form.get("Credit_History"))
        Property_Area=int(request.form.get("Property_Area"))

        features=[[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]]


        # logistic_model=logistic_model.predict(features)
        new_data_scaled=standard_scaler.transform(features)
        result=logistic_model.predict(new_data_scaled)
        print(result)

        if result[0] == 1:
          result = "Congratulations! Your loan has been approved."
        else:
           result = "Sorry, your loan application was not approved."


        
        return render_template("home.html", prediction_text="This User Will get {}".format(result))

        
        
    else:
        return render_template("home.html")
    


if __name__ == "__main__":
    app.run(debug=True)
