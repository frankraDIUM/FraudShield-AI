import pandas as pd
import requests
import time

# Load the bridge file
df_test = pd.read_csv('test_data.csv')

def production_simulation():
    # Loop forever or for N transactions
    for i in range(100):
        sample = df_test.sample(1)
        # Convert all features to a list for the API
        # Drop 'Class' because the API doesn't get the answer!
        features_list = sample.drop('Class', axis=1).values.flatten().tolist()

        # Send to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict", json={"features": features_list})
        result = response.json()

        print(f"Transaction {i}: Model says {result['decision']} (Prob: {result['probability']:.4f})")
        time.sleep(1)

production_simulation()
