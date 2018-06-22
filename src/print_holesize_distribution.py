from os import listdir
from os.path import join, isfile
import json
from random import randint 

#########################################
## START of part that students may change
from code_completion_context_CNN_random_search import Code_Completion_Context_CNN
#import os

training_dir = "./../training_data/programs_800/"
query_dir = "./../training_data/programs_200/"
model_file = "./trained_model"

#training_dir = "/home/kurse/kurs00020/mj97nyma/jobs/ASDL2018/training_data/programs_800/"
#query_dir = "/home/kurse/kurs00020/mj97nyma/jobs/ASDL2018/training_data/programs_200/"
#model_file = "/home/kurse/kurs00020/mj97nyma/jobs/ASDL2018/src/trained_model"
use_stored_model = False
#store_model = True

max_hole_size = 3
simplify_tokens = True
## END of part that students may change
#########################################

def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"

# loads sequences of tokens from _tokens.json files
def load_tokens(token_dir):
    token_files = [join(token_dir, f) for f in listdir(token_dir) if isfile(join(token_dir, f)) and f.endswith("_tokens.json")]
    token_lists = [json.load(open(f, encoding='utf8')) for f in token_files]
    if simplify_tokens:
        for token_list in token_lists:
            for token in token_list:
                simplify_token(token)
    return token_lists      # list of lists, that contain token dicts of the form {'type': 'Punctuator', 'value': ';'}
                            # these lists still have the correct order

# removes up to max_hole_size tokens
def create_hole(tokens):
    hole_size = min(randint(1, max_hole_size), len(tokens) - 1)
    hole_start_idx = randint(1, len(tokens) - hole_size)
    prefix = tokens[0:hole_start_idx]
    expected = tokens[hole_start_idx:hole_start_idx + hole_size]
    suffix = tokens[hole_start_idx + hole_size:]
    return(prefix, expected, suffix)

# checks if two sequences of tokens are identical
def same_tokens(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        return False
    for idx, t1 in enumerate(tokens1):
        t2 = tokens2[idx]
        if t1["type"] != t2["type"] or t1["value"] != t2["value"]:
            return False  
    return True

#########################################
## START of part that students may change
#code_completion = Code_Completion_Context_CNN()
## END of part that students may change
#########################################

# train the network
#training_token_lists = load_tokens(training_dir)
#if use_stored_model:
#    code_completion.load(training_token_lists, model_file)
#else:
#    code_completion.train(training_token_lists, model_file)

# query the network and measure its accuracy
query_token_lists = load_tokens(query_dir)
correct = incorrect = 0
counts = dict()
counts[1] = 0
counts[2] = 0
counts[3] = 0
for tokens in query_token_lists:
    (prefix, expected, suffix) = create_hole(tokens)
    print("expected=%s"%expected)
    counts[len(expected)] += 1
for l in counts.keys():
    print("Words of length%d: %d"%(l, counts[l]))
    #completion = code_completion.query(prefix, suffix)
    #if same_tokens(completion, expected):
    #    correct += 1
    #else:
    #    incorrect += 1
#accuracy = correct / (correct + incorrect)
#code_completion.model.save(model_file)      # later on this should only happen if the current model is better than the previous best one
#print("Accuracy: " + str(correct) + " correct vs. " + str(incorrect) + " incorrect = "  + str(accuracy))
