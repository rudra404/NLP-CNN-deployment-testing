from flask import Flask, request
import torch
import torchtext.transforms as T
import logging
import json
from model import CNN2
from tokeniser import SpacyTokenizer
#from memory_profiler import profile

#Setting up logging
logging.basicConfig(filename='service.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)

#Start logger
log = logging.getLogger('werkzeug')
log.disabled = True

#Labels for the different sentiments
LABELS = ['amusement', 'anger', 'approval', 'confusion', 'curiosity',
       'desire', 'disapproval', 'fear', 'gratitude', 'joy', 'love',
       'remorse', 'sadness', 'neutral']
    
#If GPU is available use cude
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load in vocab
text_vocab = torch.load('vocab_obj.pt')

model = CNN2()
model.to(DEVICE)
#Load pretrained parameters for CNN
model.load_state_dict(torch.load('NLP_model.pt', map_location=torch.device('cpu')))

#Function for predicting sentiments
def predict_sentiment(model, sentence, min_len = 5):
    #Set model to evaluation mode
    model.eval()
    #Tokenise sentence
    tokenizer = SpacyTokenizer()
    tokenized = tokenizer(sentence)

    #If sentence is too small for CNN pad it
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    
    #Transform tokens and convert to tensor for CNN
    vocab_transform = T.VocabTransform(text_vocab)
    indexed = vocab_transform(tokenized)
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(0)

    #Make prediction with CNN
    prediction = torch.max(model(tensor), dim=1)
    return LABELS[prediction[1].item()]



#app route to check flask app is running
@app.route('/')
def hello_world():
   return 'App is running'

#App route for getting sentiment from one sentence
@app.route('/get_sentiment',  methods=['POST'])
#@profile
def predict():
    if request.method == 'POST':
        #Get text from json
        text = request.get_json()
        #predict sentiment
        sentiment = predict_sentiment(model, sentence=text)
        #Log prediction
        logging.info(f'User input: {text}, Model prediction: {sentiment}')

        return json.dumps(sentiment)
        
#App route for retrieving sentiments from multiple sentences
@app.route('/retreive_sentiments',  methods=['POST'])
def testHandler():
    if request.method == 'POST':
        #Get text from json
        text = json.loads(request.get_json())
        #predict sentiment
        predictions = []
        for sentence in text:
            sentiment = predict_sentiment(model, sentence=sentence)
            predictions.append(sentiment)
        #Log prediction
        logging.info(f'User input: {text}, Model prediction: {predictions}')

        return json.dumps(predictions)

#Run app on local host 
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5050, threaded=True, debug=True)
