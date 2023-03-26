from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import json
import pickle

with open('artifacts/project_data.json','r') as file:
    project_data = json.load(file)

with open('artifacts\model.pkl','rb') as file:
    model = pickle.load(file)

app= Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods = ['POST'])
def get_data():
    data = request.form
    PG_Count= data["html_pages"]
    GSM = data["html_gsm"]
    Binding = data["html_binding"]
    Paper= data["html_paper"]

    user_data = np.zeros(len(project_data['column_names']))
    user_data[0] = PG_Count
    user_data[1] = GSM
    user_data[2] = project_data['Binding '][Binding]
    
    search_paper = 'Paper_'+Paper
    index = np.where(np.array(project_data['column_names']) == search_paper)[0][0]
    user_data[index] = 1

    result = model.predict([user_data])
    print(result)
    return render_template('index.html', prediction = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
