import logging
import os
import pickle
import json

import azure.functions as func

ANOMALY = -1
current_dir = os.path.abspath(__file__)
scriptdir = os.path.dirname(current_dir)
model_file = os.path.join(scriptdir, "model.sav")
scaler_file = os.path.join(scriptdir, "scaler.sav")

#load model & scaler
model = pickle.load(open(model_file, 'rb'))
scaler = pickle.load(open(scaler_file, 'rb'))

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    i = 0
    # I imagine you could replace this line with
    # events = req.get_json() to get the HTTP request body of data
    events = get_data()
    transformed_data = scaler.transform(events)
    classification = model.predict(transformed_data)

    anomaly = []
    normal = []

    #print event data with labels 
    for p in classification:
        if p == ANOMALY:
            #TODO send data to data visualization service 
            print("Anomaly: ", events[i]) #placeholder
            anomaly.append(events[i])
        else:
            print("Normal: ", events[i]) #placeholder
            normal.append(events[i])
        i += 1

    return func.HttpResponse(json.dumps({ 'anamoly': anomaly, 'normal': normal}))

def get_data():
    #TODO receive data from IoT device 
    return [[-2.81e+00, -8.38e+00, -8.69e+00,  1.69e+00,  2.28e+00,  1.00e-02,  8.41e+02], [-8.500e+00,  2.500e+00, -3.470e+00,  2.400e-01,  3.100e+00, -4.200e-01, -4.175e+03], [ 6.32e+00,  1.32e+00,  7.20e+00,  1.10e-01, -1.70e-01, -6.00e-02, -4.79e+02]] #placeholder
