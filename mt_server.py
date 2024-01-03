#!/bin/python
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
# from utils.mt import multilingual as mt # THIS IS AN EXAMPLE; HERE LOAD THE TRANSLATION SCRIPT
from flask_cors import CORS
import json

import sys
import torch
import numpy as np
from utils import initialise, read_args
from translate_sentence import translate_sentence

# creating the flask app
app = Flask(__name__)
api = Api(app)
CORS(app, origins="http://localhost:8080", allow_headers=[
    "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    supports_credentials=True)

global mt

# a class for resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class mt_sent(Resource):
    ''' MT on sentence level
    '''
    def __init__(self):
        self.args = read_args(sys.argv[1])
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 
        np.random.seed(self.args.seed)      
        torch.cuda.manual_seed_all(self.args.seed) 
        (
            _,
            self.model,
            self.accelerator
        ) = initialise(self.args, self.args.evaluation_mode, 'last', multigpu=False)
        self.model.eval()
        self.lang_mapping = {
            'spa': 'es_XX',
            'eng': 'en_XX',
            'gle': 'ga_GA',
            'nld': 'nl_XX',
            'vgt': 'vgt', # Flemish SL
            'ssp': 'ssp', # Spanish SL
            'bfi': 'bfi', # British SL
            'dse': 'dse' # British SL
        }

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def post(self):
        if request.method == 'POST':
            json_data = request.get_json(force = True)
            length = len(json_data)-1

            txt = json_data['src_text'] if length > 1 else "null"
            src_lang = json_data['src_lang'] if length > 1 else "null"
            trg_lang = json_data['trg_lang']  if length > 1 else "null"
 
            #translation = mt.translate(txt, src_lang, trg_lang) # UNCOMMENT FOR TRANSLATION
            translation = translate_sentence(
                self.args,
                self.model,
                self.accelerator,
                self.model.tokeniser,
                txt,
                self.lang_mapping[src_lang.lower()],
                self.lang_mapping[trg_lang.lower()]
            )
            #translation = "This is not a test" # Change this accordingly

        return(jsonify({
            'translation':{'src_text': txt, 'mt_text': translation}
        }))


# passing MT and Ref for quality evaluation (sentence-level)
api.add_resource(mt_sent, '/mt_sent')

@app.route("/")
def home():
    return "MT server for SignON: <h1>API</h1>"

# driver function
if __name__ == '__main__':
    app.run(debug = True)
