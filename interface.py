from flask import Flask, render_template, request, redirect, url_for, jsonify
from utils import Titanic

app = Flask(__name__)

@app.route('/')
def home():
    print("Going To Home Page...!")
    return render_template("home.html")

@app.route('/titanic', methods = ['GET','POST'])
def get_predict_titanic():
    if request.method == 'GET':

        Pclass = eval(request.args.get('Pclass'))
        Gender = request.args.get('Gender')
        Age    = eval(request.args.get('Age'))
        SibSp  = eval(request.args.get('SibSp'))
        Parch  = eval(request.args.get('Parch'))
        Fare   = eval(request.args.get('Fare')) 
        Embarked = request.args.get('Embarked')

        print("Pclass, Gender, Age, SibSp, Parch, Fare, Embarked:\n",Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)

        titanic = Titanic(Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)
        predict = titanic.predict_model()
        # return jsonify({"Prediction:":int(predict)})
        return render_template("home.html",prediction = int(predict))


if __name__  == "__main__":
    app.run(host='0.0.0.0', port=5055)   