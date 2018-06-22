import tflearn
import numpy

class Code_Completion_Baseline:

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
    
    """
    gets a set of tokens_lists. Each list represents a program.
    Initializes dicts, that map unique numbers to each KIND of token. I.e.: 50 different kinds of tokens -> each has unique number
    Those are used to utilize one hot encoding with appropriate length.
    X Y pairs are created, consisting of one hot encoded vectors.
    :return: (x_list, y_list)
    """
    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
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
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))


        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    # creates the neural network
    def create_network(self):
        # define input_data layer: shape tells us how the input data looks like. First element defines batch size and should be "None"
        self.net = tflearn.input_data(shape=[None, len(self.string_to_number)])
        # add a deep layer with 32 nodes (and linear activation function?)
        self.net = tflearn.fully_connected(self.net, 32)
        # add a deep layer with softmax activation and as many nodes as one hot encoding length
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        # output with a regression layer
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)
    
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True)
        self.model.save(model_file)
        
    # gets a hole in the code, a.k.a prefix and suffix and predicts what has to be in the hole
    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        y = self.model.predict([x])
        predicted_seq = y[0]
        #print("PREDICTION:")
        #print(y)
        #print("=================")
        #print(predicted_seq)
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        # return prediction as a list, containing the predicted token as a dict
        return [best_token]
    
