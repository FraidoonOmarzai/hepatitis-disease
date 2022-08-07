from flask import Flask, render_template, url_for, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("hepatitis.html")

@app.route('/predictHD', methods=["POST"])
def predictHD():

    loaded_model = joblib.load('Trained Model/hepatitis_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))

if __name__ == "__main__":
    app.run(debug=True)
