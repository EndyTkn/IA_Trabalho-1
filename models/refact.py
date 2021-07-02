import pandas as pd
import math

def replace_sex(sex):
    if (sex == 'm'):
        return 0
    return 1

def replace_nan(value):
    if (isinstance(value, str)):
        return value
    if (math.isnan(value)):
        return 0
    return value

def refact_dataset(input, output):
    hepatite = pd.read_csv(input)

    hepatite["Sex"] = hepatite["Sex"].apply(replace_sex)
    for category in hepatite:
        hepatite[category] = hepatite[category].apply(replace_nan)

    hepatite.to_csv(output)
