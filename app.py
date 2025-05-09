from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])  # 0 = female, 1 = male
        age = float(request.form['age'])
        fare = float(request.form['fare'])

        input_data = np.array([[pclass, sex, age, fare]])
        prediction = model.predict(input_data)[0]
        message = "Sobrevivió" if prediction == 1 else "No sobrevivió"
        return render_template('index.html', prediction_text=message)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)