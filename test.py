import copy
import json
import random

import requests

BASE_URL = "http://127.0.0.1:7000"

REQUIRED_FIELDS = [
    "Env_Temp",
    "BMP_Pressure",
    "BMP_Altitude",
    "Light",
    "DHT_Humidity",
    "MQ135_CO2",
    "MQ135_CO",
    "MQ135_NH3",
    "Soil_Moisture",
    "Soil_Temp",
    "pH",
    "N",
    "P",
    "K",
    "district",
    "location",
    "season"
]

VALID_SAMPLE = {
    "Env_Temp": 28.5,
    "BMP_Pressure": 980.2,
    "BMP_Altitude": 120.0,
    "Light": 35000,
    "DHT_Humidity": 78.0,
    "MQ135_CO2": 420.0,
    "MQ135_CO": 2.5,
    "MQ135_NH3": 1.1,
    "Soil_Moisture": 45.0,
    "Soil_Temp": 22.5,
    "pH": 6.5,
    "N": 240.0,
    "P": 18.0,
    "K": 210.0,
    "district": "Raigad",
    "location": "Karjat",
    "season": "Monsoon"
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def post_predict(payload, description):
    print(f"\n=== TEST: {description} ===")
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        print("Status Code:", r.status_code)
        print("Response:", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("REQUEST FAILED:", e)


# --------------------------------------------------
# TESTS
# --------------------------------------------------
def test_server_alive():
    print("\n=== TEST: Server Alive ===")
    r = requests.get(f"{BASE_URL}/")
    print("Status Code:", r.status_code)


def test_valid_prediction():
    post_predict(VALID_SAMPLE, "Valid input (happy path)")


def test_missing_single_field():
    data = copy.deepcopy(VALID_SAMPLE)
    data.pop("N")
    post_predict(data, "Missing single required field (N)")


def test_missing_multiple_fields():
    data = copy.deepcopy(VALID_SAMPLE)
    data.pop("N")
    data.pop("P")
    data.pop("K")
    post_predict(data, "Missing multiple required fields (N, P, K)")


def test_extra_unused_field():
    data = copy.deepcopy(VALID_SAMPLE)
    data["random_noise"] = 999
    post_predict(
        data, "Extra unused field (should be ignored or rejected cleanly)")


def test_wrong_datatype_numeric():
    data = copy.deepcopy(VALID_SAMPLE)
    data["N"] = "two hundred"
    post_predict(data, "Wrong datatype for numeric field (N as string)")


def test_null_values():
    data = copy.deepcopy(VALID_SAMPLE)
    data["P"] = None
    post_predict(data, "Null value in numeric field (P=None)")


def test_all_null_numerics():
    data = copy.deepcopy(VALID_SAMPLE)
    for k in data:
        if k not in ["district", "location", "season"]:
            data[k] = None
    post_predict(data, "All numeric fields null")


def test_extreme_values():
    data = copy.deepcopy(VALID_SAMPLE)
    data.update({
        "Env_Temp": -50,
        "BMP_Pressure": 2000,
        "Light": 1e7,
        "N": 10000,
        "P": -100,
        "K": 99999
    })
    post_predict(data, "Extreme out-of-range numeric values")


def test_invalid_category():
    data = copy.deepcopy(VALID_SAMPLE)
    data["district"] = "Mars"
    post_predict(data, "Invalid categorical value (district=Mars)")


def test_case_sensitivity():
    data = copy.deepcopy(VALID_SAMPLE)
    data["season"] = "monsoon"  # lowercase
    post_predict(data, "Case sensitivity in categorical values")


def test_empty_payload():
    post_predict({}, "Empty JSON payload")


def test_malformed_json():
    print("\n=== TEST: Malformed JSON ===")
    headers = {"Content-Type": "application/json"}
    bad_json = "{this is not valid json}"
    r = requests.post(f"{BASE_URL}/predict", data=bad_json, headers=headers)
    print("Status Code:", r.status_code)
    print("Response:", r.text)


def test_random_fuzzing(iterations=5):
    for i in range(iterations):
        data = {}
        for field in REQUIRED_FIELDS:
            if field in ["district", "location", "season"]:
                data[field] = random.choice(
                    ["Raigad", "Nashik", "Invalid", "", None]
                )
            else:
                data[field] = random.choice(
                    [random.uniform(-100, 1000), None, "bad", 0]
                )

        post_predict(data, f"Random fuzz test #{i+1}")


def test_google_sheets_endpoint():
    print("\n=== TEST: Google Sheets endpoint ===")
    r = requests.get(f"{BASE_URL}/latest-readings?ID=1")
    print("Status Code:", r.status_code)
    try:
        print("Response:", json.dumps(r.json(), indent=2))
    except Exception:
        print("Non-JSON response:", r.text)


# --------------------------------------------------
# RUN ALL TESTS
# --------------------------------------------------
if __name__ == "__main__":
    test_server_alive()
    test_valid_prediction()
    test_missing_single_field()
    test_missing_multiple_fields()
    test_extra_unused_field()
    test_wrong_datatype_numeric()
    test_null_values()
    test_all_null_numerics()
    test_extreme_values()
    test_invalid_category()
    test_case_sensitivity()
    test_empty_payload()
    test_malformed_json()
    test_random_fuzzing(iterations=5)
    test_google_sheets_endpoint()
