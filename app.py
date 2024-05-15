from flask import Flask, render_template, url_for ,request
# from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
import random

from static.breast_cancer import X_test, Y_test
app = Flask(__name__)
# db=SQLAlchemy(app)
# app.config['SQLACHEMY_DATABASE_URI']='sqlite:///database.db'
# app.config['SECRET_KEY']='thisisasecretkey'


import pickle
model=pickle.load(open("models/breast_cancer.pkl","rb"))
model2=pickle.load(open("models/project.pkl","rb"))
with open("models/kidneyy.pkl", "rb") as f:
    model3 = pickle.load(f)
with open("models/model4.pkl", "rb") as f:
    model4= pickle.load(f)
model5=pickle.load(open("models/insurance.pkl","rb"))
import numpy as np

#create a tabel



@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/options')
def options():
    return render_template('bc_index4.html')

@app.route('/form1')
def form1():
    return render_template('bc_index2.html')

@app.route('/form')
def form():
    return render_template('bc_index3.html')

@app.route('/openn')
def openn():
    return render_template('hd_index2.html')

@app.route('/hello')
def hello():
    return render_template('kidney_index2.html')

@app.route('/openit')
def openit():
    return render_template('diabetes_index.html')

@app.route('/insurance')

def insurance():
    return render_template('insurance.html')

@app.route('/predict6', methods=['POST','GET'])
def predict6():
    # features = [float(x) for x in request.form.values()]

    features= request.form.to_dict()
    data_array = np.array(list(features.values()), dtype=float)
    data_array=data_array.reshape(1, -1)  
    prediction = model5.predict(data_array).item()
    # print(features)
    # final = np.array(features).reshape((1,6))
    # print(final)
    # pred = model5.predict(final)[0]
    # print(pred)

    
    if prediction < 0:
        return render_template('amount.html', prediction='Error calculating Amount!')
    else:
        formatted_pred = 'Expected amount is {0:.3f}'.format(prediction)
        return render_template('amount.html',prediction=formatted_pred)
        # return render_template('amount.html', prediction='Expected amount is {0:.3f}'.format(prediction))
    
    
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_array = np.array(list(form_data.values()), dtype=float)
        data_array=data_array.reshape(1, -1)  
        prediction = model.predict(data_array)
        n=""
        # Return the prediction
        if(prediction[0]==0):
            return render_template('B.html')
        else:
            return render_template('B.html')
    
        # return render_template('bc_index2.html', prediction=n)
        

@app.route('/predict2', methods=['POST'])
def predict2():
    if request.method == 'POST':
        # Extract input data from form
        csv_input = request.form['csv-input']
        
        # Convert comma-separated string to array of floats
        features = list(map(float, csv_input.split(',')))
        data_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data_array)
        if(prediction[0]==0):
            return render_template('M.html')
        else:
            return render_template('B.html')
        
        
@app.route('/predict3', methods=['POST'])
def predict3():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        # expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        #                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        # if not all(feature in form_data for feature in expected_features):
        #     return 'Error: Form data is missing some features.'
        data_array = np.array(list(form_data.values()), dtype=float)
        data_array=data_array.reshape(1, -1)  
        prediction = model2.predict(data_array)
        n=""
        # Return the prediction
        if(prediction[0]==0):
            return render_template('nohd.html')
        else:
            return render_template('hd.html')
       
@app.route('/predict4', methods=['POST'])
def predict4():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_array = np.array(list(form_data.values()), dtype=float)
        data_array=data_array.reshape(1, -1)  
        prediction = model3.predict(data_array)
        n=""
        # Return the prediction
        if(prediction[0]==0):
            return render_template('nokidney.html')
        else:
            return render_template('kidney.html')

@app.route('/predict5', methods=['POST'])
def predict5():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_array = np.array(list(form_data.values()), dtype=float)
        data_array=data_array.reshape(1, -1)  
        prediction = model4.predict(data_array)
        n=""
        # Return the prediction
        if(prediction[0]==0):
            return render_template('nodiabetes.html')
        else:
            return render_template('diabetes.html')
       
if __name__ == '__main__':
    app.run(debug=True)
    