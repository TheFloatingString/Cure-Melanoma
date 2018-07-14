"""
Toolkits for basic ML usage for the project
"""

import json

def constant(x):
    """Constant function"""
    return x


ref_treatment = {"Melanoma" : "If you have melanoma skin cancer, your healthcare team will create a treatment plan just for you. It will be based on your health and specific information about the cancer.",
                "Melanocytic Nevus": "If it is detected in later stages, it will be removed, plus some healthy skin around it - called a safety margin. If the cancer has entered the bloodstream or lymphatic system and formed tumors in other parts of the body, the patient will require further treatment."}

with open('ref_treatment.json', 'w') as fp:
    json.dump(ref_treatment, fp)
