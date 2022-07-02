from flask import Flask, request, render_template
import pickle

vectorizer = pickle.load(open("vectorizer.pkl", 'rb')) 
model = pickle.load(open("regression_model.pkl", 'rb'))
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        news=vectorizer.transform([news])
        predict = model.predict(news)

        return render_template("prediction.html", result= "Great !!!Looks Like The News is Real" if predict == 0 else "Oh No !!! Looks Like The News is Fake")
    else:
        return render_template("prediction.html")
    

if __name__ == '__main__':
    app.run()