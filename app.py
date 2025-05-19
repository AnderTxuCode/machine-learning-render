from flask import Flask, request, render_template
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

try:
    model = joblib.load("./models/decision_tree_classifier_default_42.sav")
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            val1 = float(request.form["val1"])
            val2 = float(request.form["val2"])
            val3 = float(request.form["val3"])
            val4 = float(request.form["val4"])
            
            data = [[val1, val2, val3, val4]]
            
            prediction = str(model.predict(data)[0])
            pred_class = class_dict.get(prediction, "Clase desconocida")
        except Exception as e:
            pred_class = f"Error al predecir: {str(e)}"
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)