from flask import Flask,render_template,jsonify,request
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

app = Flask(__name__)

@app.route("/")
def home_page():
    return "Welcome to Adult Census Salary Prediction"

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        request_data = request.get_json()
        # for key,value in request_data.items():
        data = CustomData(
            age=request_data['age'],
            occupation=request_data['occupation'],
            workclass = request_data['workclass'],
            fnlwgt = request_data['fnlwgt'],
            education_num = request_data['education_num'],
            marital_status = request_data['marital_status'],
            relationship = request_data['relationship'],
            race = request_data['race'],
            sex = request_data['sex'],
            capital_gain = request_data['capital_gain'],
            capital_loss = request_data['capital_loss'],
            hours_per_week=request_data['hours_per_week'],
            country=request_data['country'],
            education= request_data['education']
        )
        # data = CustomData(
        # carat = float(request.form.get("carat")),
        # depth = float(request.form.get("depth")),
        # table = float(request.form.get("table")),
        # x = float(request.form.get("x")),
        # y = float(request.form.get("y")),
        # z = float(request.form.get("z")),
        # cut = request.form.get("cut"),
        # color = request.form.get("color"),
        # clarity = request.form.get("clarity")
        # )
    
        final_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)
        result = pred[0,0]
        # result = round(pred[0,0],2)
        return {
            'predicted_salary':result
        }

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80)
