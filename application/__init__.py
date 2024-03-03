from flask import Flask
from flask_cors import CORS
from joblib import load
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import requests


app=Flask(__name__)
CORS(app)
scheduler = BackgroundScheduler()

loaded_pred_maintenance_model = load('models/predictive_maintenance_model.joblib')
rf_classifier = load('models/rf_model.pkl')
tfidf_vectorizer = load('models/tfidf_vectorizer.pkl')

loaded_rf_classifiers = {}
attributes = ['diet', 'flavor_profile', 'course', 'state', 'region']
for attribute in attributes:
    classifier = load(f'models/{attribute}_classifier.joblib')
    feature_selector = load(f'models/{attribute}_feature_selector.joblib')
    loaded_rf_classifiers[attribute] = (classifier, feature_selector)

# Load the TfidfVectorizer
loaded_vectorizer = load('models/vectorizer.joblib')

from application import routes