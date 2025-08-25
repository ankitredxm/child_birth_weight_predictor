from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle

app=Flask(__name__)


## define your endpoint



def get_cleaned_data(form_data):
    gestation=float(form_data['gestation'])
    parity=int(form_data['parity'])
    age=float(form_data['age'])
    height=float(form_data['height'])
    weight=float(form_data['weight'])
    smoke=float(form_data['smoke'])

    cld={
        "gestation":[gestation],
        "parity":[parity],
        "age":[age],
        "height":[height],
        "weight":[weight],
        "smoke":[smoke]
    }
    return cld





@app.route("/",methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def get_prediction():
    # get data from user
    baby_data=request.form
    baby_cleaned_data=get_cleaned_data(baby_data)

    # convert into data frame
    baby_df=pd.DataFrame(baby_cleaned_data)

    # load machine l trained model
    with open("model.pkl","rb") as obj:
        model=pickle.load(obj)

    # make prediction on user data
    baby_prediction=model.predict(baby_df)

    prediction=round(float(baby_prediction),2)

    #return response 

    

    return render_template("index.html",prediction=prediction)













if __name__=='__main__':
    app.run(debug=True)