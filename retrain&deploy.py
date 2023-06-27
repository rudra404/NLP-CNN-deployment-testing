import subprocess
import json
import requests
import time

# Run train.py
train_process = subprocess.Popen(['python', 'train.py'])

# Wait for the train.py process to complete
train_process.wait()

# Run train.py
test_process = subprocess.Popen(['python', 'test.py'])

# Wait for the train.py process to complete
train_process.wait()

# Run app.py
Deploy_app = subprocess.Popen(['python', 'app.py'])

time.sleep(1)

#Test get_sentiment API call
url = "http://127.0.0.1:5050/get_sentiment"
data = "I'm so upset, I have been crying all day"
payload = json.dumps(data)  # Convert the list of comments to JSON format
response = requests.post(url, json=payload)
assert response.status_code == 200, "Get sentiment request failed"
assert isinstance(response.text, str), "Get sentiment does not return a string as expected"

#Test retrieve_sentiments API call
url2 = "http://127.0.0.1:5050/retreive_sentiments"
data2 = ["Yay another day of work, I'm sooooo excited (not)", "I am angry"]
payload2 = json.dumps(data2)  # Convert the list of comments to JSON format
response = requests.post(url2, json=payload2)
assert response.status_code == 200, "Retrieve sentiments request failed"
assert isinstance(response.text, str), "Retrieve sentiments does not return a string of sentiments as expected"

print("Flask app working as expected")