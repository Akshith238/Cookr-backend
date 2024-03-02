from application import app, rf_classifier,tfidf_vectorizer,loaded_pred_maintenance_model
from flask import jsonify, request
import numpy as np
import pandas as pd

def generate_live_data():
    cpu_usage = float(np.random.uniform(0, 100))
    memory_utilization = float(np.random.uniform(0, 100))
    network_traffic_in = float(np.random.uniform(0, 1000))
    network_traffic_out = float(np.random.uniform(0, 1000))
    
    input_data = {
        'cpu_usage': [cpu_usage],
        'memory_utilization': [memory_utilization],
        'network_traffic_in': [network_traffic_in],
        'network_traffic_out': [network_traffic_out]
    }
    
    return input_data



@app.route("/api/predictFood", methods=["POST"])
def foodPredict():
    try:
        data = request.json
        dish_name = data["dish_name"] 
        dish_name_encoded = tfidf_vectorizer.transform([dish_name])
        prediction = rf_classifier.predict(dish_name_encoded)
        return jsonify(prediction.tolist()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predictFailure", methods=["GET"])
def receive_data():
    try:
        input_data_dict = generate_live_data()
        input_data_df = pd.DataFrame(input_data_dict)
        probabilities = loaded_pred_maintenance_model.predict_proba(input_data_df)
        threshold = 0.5
        failure_probability = probabilities[0][1]
        # Perform prediction using your loaded model
        alert = None
        if failure_probability > threshold:
            alert = 'Alert! Probability of failure exceeds threshold.'
        response = {
            'liveData': input_data_dict,
            'failureProbability': failure_probability,
            'alert': alert
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

