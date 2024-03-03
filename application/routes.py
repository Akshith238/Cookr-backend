from application import app, rf_classifier,tfidf_vectorizer,loaded_rf_classifiers,loaded_vectorizer,loaded_pred_maintenance_model
from flask import jsonify, request
import numpy as np
import pandas as pd
import psutil


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

def bytes_to_mb(bytes_value):
    return bytes_value / (1024)


def generate_live_data1():
    cpu_usage = [psutil.cpu_percent()]
    memory_utilization = [psutil.virtual_memory().percent]
    network_info = psutil.net_io_counters()
    
    # Scale the network traffic values between 0 and 1000
    network_traffic_in = [float(bytes_to_mb(network_info.bytes_recv))]
    network_traffic_out = [float(bytes_to_mb(network_info.bytes_sent))]
    
    input_data = {
        'cpu_usage': cpu_usage,
        'memory_utilization': memory_utilization,
        'network_traffic_in': network_traffic_in,
        'network_traffic_out': network_traffic_out
    }
    
    return input_data



def predict_attributes(dish_name, dish_ingredients):
    dish_combined = ' '.join([dish_name, dish_ingredients])
    dish_vectorized = loaded_vectorizer.transform([dish_combined])
    predicted_attributes = {}
    for attribute, (classifier, feature_selector) in loaded_rf_classifiers.items():
        selected_features = feature_selector.transform(dish_vectorized)
        predicted_attribute = classifier.predict(selected_features)[0]
        if predicted_attribute == "-1":
            predicted_attribute = "Unknown"
        predicted_attributes[attribute] = predicted_attribute
    # Check for non-vegetarian ingredients
    non_veg_ingredients = ["chicken", "mutton", "beef", "pork", "fish", "egg", "prawn", "lobster", "crab"]
    if any(ing.lower() in dish_ingredients.lower() for ing in non_veg_ingredients):
        predicted_attributes['diet'] = 'non vegetarian'
    return predicted_attributes



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

@app.route("/api/predictFood2", methods=["POST"])
def foodPredict2():
    try:
        # Parse request JSON
        data = request.json
        dish_name = data["dish_name"]
        dish_ingredients = data["dish_ingredients"]
        predicted_attributes = predict_attributes(dish_name, dish_ingredients)
        return jsonify(predicted_attributes), 200
    
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

