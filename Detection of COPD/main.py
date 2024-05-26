from flask import Flask, request, render_template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pickle
from numpy import *

app = Flask(__name__,static_folder='static')


with open('/Users/knithin/Desktop/Capstone_2020_23150_2/xgboost_model.pkl', 'rb') as file:
    xg_model = pickle.load(file)


with open('/Users/knithin/Desktop/Capstone_2020_23150_2/LightGBM_model.pkl', 'rb') as file:
    lg_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('/index.html')

mg = -1

@app.route('/action_xg/')
def action1():
    global mg
    mg = 0
    return render_template('/open.html')

@app.route('/action_lg/')
def action2():
    global mg
    mg = 1
    return render_template('/open.html')

g_out = ""

@app.route('/process_input/', methods=['POST'])
def process_input():

    gender_encoding = {
        'Female': 0,
        'Male': 1
    }

    asthma_encoding = {
        'No': 0,
        'Yes': 1
    }

    bronchitis_attack_encoding = {
        'No': 0,
        'Yes': 1
    }

    pneumonia_encoding = {
        'No': 0,
        'Yes': 1
    }

    chronic_bronchitis_encoding = {
        'No': 0,
        'Yes': 1
    }

    emphysema_encoding = {
        'No': 0,
        'Yes': 1
    }

    sleep_apnea_encoding = {
        'No': 0,
        'Yes': 1
    }

    smoking_status_encoding = {
        'Current Smoker': 0,
        'Former Smoker': 1,
        'Never Smoked': 2
    }

    visit_age = float(request.form['visit_age'])
    gender = gender_encoding[request.form['gender'].title()]
    height_cm = float(request.form['height_cm'])
    weight_kg = float(request.form['weight_kg'])
    sysBP = float(request.form['sysBP'])
    diasBP = float(request.form['diasBP'])
    hr = float(request.form['hr'])
    O2_hours_day = float(request.form['O2_hours_day'])
    bmi = float(request.form['bmi'])
    asthma = asthma_encoding[request.form['asthma'].title()]
    bronchitis_attack = bronchitis_attack_encoding[request.form['bronchitis_attack'].title()]
    pneumonia = pneumonia_encoding[request.form['pneumonia'].title()]
    chronic_bronchitis = chronic_bronchitis_encoding[request.form['chronic_bronchitis'].title()]
    emphysema = emphysema_encoding[request.form['emphysema'].title()]
    sleep_apnea = sleep_apnea_encoding[request.form['sleep_apnea'].title()]
    smoking_start_age = float(request.form['SmokStartAge'])
    cigarettes_per_day_avg = float(request.form['CigPerDaySmokAvg'])
    duration_smoking = float(request.form['Duration_Smoking'])
    smoking_status = smoking_status_encoding[request.form['smoking_status'].title()]
    total_lung_capacity = float(request.form['total_lung_capacity'])
    pct_emphysema = float(request.form['pct_emphysema'])
    functional_residual_capacity = float(request.form['functional_residual_capacity'])
    pct_gastrapping = float(request.form['pct_gastrapping'])

    FEV1_FVC_ratio = float(request.form['FEV1_FVC_ratio'])
    FEV1 = float(request.form['FEV1'])
    FVC = float(request.form['FVC'])
    FEV1_phase2 = float(request.form['FEV1_phase2'])
    
    stored_variables = np.array([
        visit_age, gender, height_cm, weight_kg, sysBP, diasBP, hr, O2_hours_day, bmi, asthma,
        bronchitis_attack, pneumonia, chronic_bronchitis, emphysema, sleep_apnea, smoking_start_age,
        cigarettes_per_day_avg, duration_smoking, smoking_status, total_lung_capacity, pct_emphysema,
        functional_residual_capacity, pct_gastrapping, FEV1_FVC_ratio,
        FEV1, FVC, FEV1_phase2,
        0, 0  
    ])
    stored_variables = stored_variables.reshape(1, 29)
    
    encoded_labels = {
        0: 'Mild',
        1: 'Moderate',
        2: 'No COPD',
        3: 'Severe',
        4: 'Very Low',
        5: 'Very Severe'
    }

    global g_out
    if mg == 0:
        value = xg_model.predict(stored_variables)
        output = encoded_labels[value[0]]
        g_out = output
        if output == 'No COPD':
            return render_template('/output.html', n="No, The Person Not is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
        else:
            return render_template('/output.html', n="Yes, The Person is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)

    elif mg == 1:
        value = lg_model.predict(stored_variables)
        output = encoded_labels[value[0]]
        g_out = output
        if output == 'No COPD':
            return render_template('/output.html', n="No, The Person Not is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
        else:
            return render_template('/output.html', n="Yes, The Person is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
    else:
        return "None"

@app.route('/health/')
def health():
    global g_out
    ret_value = precautions(g_out)
    print(ret_value)
    return render_template('/health.html',
                           disease_name=g_out,
                           precaution_1=ret_value[0],
                           precaution_2=ret_value[1],
                           precaution_3=ret_value[2],
                           precaution_4=ret_value[3])

def precautions(check):
    precautions_copd = {
        'Mild': [
            "Avoid exposure to smoke and air pollution",
            "Practice breathing exercises as recommended by your healthcare provider",
            "Stay up-to-date with vaccinations, including flu and pneumonia vaccines",
            "Follow a healthy diet and maintain a healthy weight"
        ],
        'Moderate': [
            "Take prescribed medications regularly",
            "Use supplemental oxygen as prescribed",
            "Avoid respiratory irritants such as strong chemicals and dust",
            "Engage in regular physical activity within your limits"
        ],
        'No Copd': [
            "Avoid smoking and exposure to secondhand smoke",
            "Maintain good indoor air quality",
            "Attend regular check-ups with your healthcare provider",
            "Stay active and maintain a healthy lifestyle"
        ],
        'Severe': [
            "Follow your healthcare provider's treatment plan diligently",
            "Consider pulmonary rehabilitation programs",
            "Use mobility aids if necessary to conserve energy",
            "Avoid exposure to cold air and extreme weather conditions"
        ],
        'Very Low': [
            "Monitor your symptoms closely and report any changes to your healthcare provider",
            "Stay hydrated and drink plenty of fluids",
            "Use air purifiers in your home to improve indoor air quality",
            "Practice relaxation techniques to reduce stress and anxiety"
        ],
        'Very Severe': [
            "Discuss advanced treatment options with your healthcare provider",
            "Consider palliative care or hospice services if appropriate",
            "Plan for end-of-life care preferences and decisions",
            "Involve family members and caregivers in your care and support network"
        ]
    }
    return precautions_copd[check.title()]

if __name__ == '__main__':
    app.run(debug=True, port=5500)
from flask import Flask, request, render_template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pickle
from numpy import *

app = Flask(__name__)

with open('/Users/knithin/Desktop/capi/xgboost_model.pkl', 'rb') as file:
    xg_model = pickle.load(file)

with open('/Users/knithin/Desktop/capi/LightGBM_model.pkl', 'rb') as file:
    lg_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('/index.html')

mg = -1

@app.route('/action_xg/')
def action1():
    global mg
    mg = 0
    return render_template('/open.html')

@app.route('/action_lg/')
def action2():
    global mg
    mg = 1
    return render_template('/open.html')

g_out = ""

@app.route('/process_input/', methods=['POST'])
def process_input():

    gender_encoding = {
        'Female': 0,
        'Male': 1
    }

    asthma_encoding = {
        'No': 0,
        'Yes': 1
    }

    bronchitis_attack_encoding = {
        'No': 0,
        'Yes': 1
    }

    pneumonia_encoding = {
        'No': 0,
        'Yes': 1
    }

    chronic_bronchitis_encoding = {
        'No': 0,
        'Yes': 1
    }

    emphysema_encoding = {
        'No': 0,
        'Yes': 1
    }

    sleep_apnea_encoding = {
        'No': 0,
        'Yes': 1
    }

    smoking_status_encoding = {
        'Current Smoker': 0,
        'Former Smoker': 1,
        'Never Smoked': 2
    }
    visit_age = float(request.form['visit_age'])
    gender = gender_encoding[request.form['gender'].title()]
    height_cm = float(request.form['height_cm'])
    weight_kg = float(request.form['weight_kg'])
    sysBP = float(request.form['sysBP'])
    diasBP = float(request.form['diasBP'])
    hr = float(request.form['hr'])
    O2_hours_day = float(request.form['O2_hours_day'])
    bmi = float(request.form['bmi'])
    asthma = asthma_encoding[request.form['asthma'].title()]
    bronchitis_attack = bronchitis_attack_encoding[request.form['bronchitis_attack'].title()]
    pneumonia = pneumonia_encoding[request.form['pneumonia'].title()]
    chronic_bronchitis = chronic_bronchitis_encoding[request.form['chronic_bronchitis'].title()]
    emphysema = emphysema_encoding[request.form['emphysema'].title()]
    sleep_apnea = sleep_apnea_encoding[request.form['sleep_apnea'].title()]
    smoking_start_age = float(request.form['SmokStartAge'])
    cigarettes_per_day_avg = float(request.form['CigPerDaySmokAvg'])
    duration_smoking = float(request.form['Duration_Smoking'])
    smoking_status = smoking_status_encoding[request.form['smoking_status'].title()]
    total_lung_capacity = float(request.form['total_lung_capacity'])
    pct_emphysema = float(request.form['pct_emphysema'])
    functional_residual_capacity = float(request.form['functional_residual_capacity'])
    pct_gastrapping = float(request.form['pct_gastrapping'])

    FEV1_FVC_ratio = float(request.form['FEV1_FVC_ratio'])
    FEV1 = float(request.form['FEV1'])
    FVC = float(request.form['FVC'])
    FEV1_phase2 = float(request.form['FEV1_phase2'])
    
    stored_variables = np.array([
        visit_age, gender, height_cm, weight_kg, sysBP, diasBP, hr, O2_hours_day, bmi, asthma,
        bronchitis_attack, pneumonia, chronic_bronchitis, emphysema, sleep_apnea, smoking_start_age,
        cigarettes_per_day_avg, duration_smoking, smoking_status, total_lung_capacity, pct_emphysema,
        functional_residual_capacity, pct_gastrapping, FEV1_FVC_ratio,
        FEV1, FVC, FEV1_phase2,
        0, 0  
    ])
    stored_variables = stored_variables.reshape(1, 29)
    
    encoded_labels = {
        0: 'Mild',
        1: 'Moderate',
        2: 'No COPD',
        3: 'Severe',
        4: 'Very Low',
        5: 'Very Severe'
    }

    global g_out
    if mg == 0:
        value = xg_model.predict(stored_variables)
        output = encoded_labels[value[0]]
        g_out = output
        if output == 'No COPD':
            return render_template('/output.html', n="No, The Person Not is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
        else:
            return render_template('/output.html', n="Yes, The Person is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)

    elif mg == 1:
        value = lg_model.predict(stored_variables)
        output = encoded_labels[value[0]]
        g_out = output
        if output == 'No COPD':
            return render_template('/output.html', n="No, The Person Not is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
        else:
            return render_template('/output.html', n="Yes, The Person is Having COPD",
                                   p="The individual's COPD status is characterized as: " + output)
    else:
        return "None"

@app.route('/health/')
def health():
    global g_out
    ret_value = precautions(g_out)
    print(ret_value)
    return render_template('/health.html',
                           disease_name=g_out,
                           precaution_1=ret_value[0],
                           precaution_2=ret_value[1],
                           precaution_3=ret_value[2],
                           precaution_4=ret_value[3])

def precautions(check):
    precautions_copd = {
        'Mild': [
            "Avoid exposure to smoke and air pollution",
            "Practice breathing exercises as recommended by your healthcare provider",
            "Stay up-to-date with vaccinations, including flu and pneumonia vaccines",
            "Follow a healthy diet and maintain a healthy weight"
        ],
        'Moderate': [
            "Take prescribed medications regularly",
            "Use supplemental oxygen as prescribed",
            "Avoid respiratory irritants such as strong chemicals and dust",
            "Engage in regular physical activity within your limits"
        ],
        'No Copd': [
            "Avoid smoking and exposure to secondhand smoke",
            "Maintain good indoor air quality",
            "Attend regular check-ups with your healthcare provider",
            "Stay active and maintain a healthy lifestyle"
        ],
        'Severe': [
            "Follow your healthcare provider's treatment plan diligently",
            "Consider pulmonary rehabilitation programs",
            "Use mobility aids if necessary to conserve energy",
            "Avoid exposure to cold air and extreme weather conditions"
        ],
        'Very Low': [
            "Monitor your symptoms closely and report any changes to your healthcare provider",
            "Stay hydrated and drink plenty of fluids",
            "Use air purifiers in your home to improve indoor air quality",
            "Practice relaxation techniques to reduce stress and anxiety"
        ],
        'Very Severe': [
            "Discuss advanced treatment options with your healthcare provider",
            "Consider palliative care or hospice services if appropriate",
            "Plan for end-of-life care preferences and decisions",
            "Involve family members and caregivers in your care and support network"
        ]
    }
    return precautions_copd[check.title()]

if __name__ == '__main__':
    app.run(debug=True, port=5500)
