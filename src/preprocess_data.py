from os import listdir
from os.path import join, isfile
import os
import json
from random import randint
import numpy

simplify_tokens = True
context_length = 5
processed_training_data_dir = "./../training_data/processed_training_data/"
training_dir = "./../training_data/programs_800/"
query_dir = "./../training_data/programs_200/"

def load_tokens(token_dir):
    token_files = [join(token_dir, f) for f in listdir(token_dir) if isfile(join(token_dir, f)) and f.endswith("_tokens.json")]
    token_lists = [json.load(open(f, encoding='utf8')) for f in token_files]
    if simplify_tokens:
        for token_list in token_lists:
            for token in token_list:
                simplify_token(token)
    return token_lists      # list of lists, that contain token dicts of the form {'type': 'Punctuator', 'value': ';'}
                            # these lists still have the correct order

def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"

class preprocessor:

    # covnerts a token, that is a dict into a string
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    # converts a strin, that represents a token into a dict
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    # maps a string (token) to a unique ID, that corresponds to where the token appeared in the code
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def trim_to_length(self, prefix:list, suffix:list, final_length:int):
        """
        receives a list of prefix and suffix tokens and trims them to the desired length, adding padding if necessary
        """

        if len(prefix) < final_length:
            needed_padding = final_length - len(prefix)
            prefix = [{"type": "padding", "value": "padding"}] * needed_padding + prefix
        elif len(prefix) > final_length:
            prefix = prefix[-final_length:]

        if len(suffix) < final_length:
            needed_padding = final_length - len(suffix)
            suffix = suffix + [{"type": "padding", "value": "padding"}] * needed_padding
        elif len(suffix) > final_length:
            suffix = suffix[:final_length]

        return (prefix, suffix)

    def one_hot_tokenlist(self, tokenlist):
        """
        takes a list of tokens, that are encoded as dicts and returns a list of concatenated one_hot_encodings of all tokens
        """
        one_hot_encodings = []
        for token in tokenlist:
            one_hot_encodings = one_hot_encodings + self.one_hot(self.token_to_string(token))

        return one_hot_encodings

    def process_and_store_tokens(self, token_lists):
        print("prepare data call")
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings.add("padding-@@-padding") # add a padding token for corners
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict() 
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1
        
        # prepare x,y pairs
        xs = []
        ys = []
        #print("xs, ys init")
        # for efficiency reasons: load data that already is in the right format instead of processing it every time
        if os.path.isfile(processed_training_data_dir + "data_ys_context_" + str(context_length) + ".npz"):
            print("loading training data from .npy files...")
            xs_dict = numpy.load(processed_training_data_dir + "data_xs_context_" + str(context_length) + ".npz")
            ys_dict = numpy.load(processed_training_data_dir + "data_ys_context_" + str(context_length) + ".npz")
            xs = xs_dict[xs_dict.keys()[0]].tolist()
            ys = ys_dict[ys_dict.keys()[0]].tolist()

        else:
            for token_list in token_lists:
                for idx, token in enumerate(token_list):
                    token_string = self.token_to_string(token)

                    # get prefix and suffix
                    prefix = token_list[:idx]
                    if idx+1 < len(token_list):
                        suffix = token_list[idx+1:]
                    else:
                        suffix = []

                    # get one_hot_encodings
                    prefix, suffix = self.trim_to_length(prefix, suffix, context_length)

                    # create data for NN
                    xs.append(self.one_hot_tokenlist(prefix) + self.one_hot_tokenlist(suffix))
                    ys.append(self.one_hot(token_string))

            # save data for later retraining
            print("saving training data to .npy files...")
            numpy.savez_compressed(processed_training_data_dir + "data_xs_context_" + str(context_length) + ".npz", numpy.array(xs))
            numpy.savez_compressed(processed_training_data_dir + "data_ys_context_" + str(context_length) + ".npz", numpy.array(ys))
            print("save done")

        return (xs, ys)

token_lists = load_tokens(training_dir)
processor = preprocessor()
xs, ys = processor.process_and_store_tokens(token_lists)