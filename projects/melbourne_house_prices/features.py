import numpy as np
import pandas as pd


NUMERIC = [
    "Rooms", "Distance", "Bedroom2", "Bathroom", "Car", "Landsize",
    "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
]
TYPE_CODES = {"h": 0, "t": 1, "u": 2}


def prepare_features(homes):
    features = pd.DataFrame(index=homes.index)
    for name in NUMERIC:
        if name in homes:
            features[name] = pd.to_numeric(homes[name], errors="coerce")
        else:
            features[name] = np.nan
    if "Type" in homes:
        features["PropertyType"] = homes["Type"].map(TYPE_CODES)
    else:
        features["PropertyType"] = np.nan
    return features


def suburb_baseline(homes, suburb_medians, overall_median):
    if "Suburb" not in homes:
        return np.full(len(homes), overall_median)
    names = homes["Suburb"].str.strip().str.casefold()
    return names.map(suburb_medians).fillna(overall_median).to_numpy()
