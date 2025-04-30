from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Route to serve the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the POST request
    data = request.get_json()
    comment = data.get("text")
    
    # Example dummy model for demonstration
    if "bad" in comment.lower():  
        prediction = 1  # Toxic
    else:
        prediction = 0  # Non-toxic
    
    # Return the prediction as JSON response
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
