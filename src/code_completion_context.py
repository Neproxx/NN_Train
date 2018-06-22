import tflearn
import numpy as np

# determines how many tokens before and after are to be used during training
context_length = 5
processed_training_data_dir = "./../training_data/processed_training_data/"

class code_completion_context:

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
    gets a set of tokens_lists. Each list represents a program with tokens in the correct order. CHANGE REPRESENTATION HERE
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

        # for efficiency reasons: load data that already is in the right format instead of processing it every time
        if os.path.isfile(processed_training_data_dir + "data_ys_context_" + str(context_length) + ".npz"):
            print("loading training data from .npy files...")
            xs_dict = numpy.load(processed_training_data_dir + "data_xs_context_" + str(context_length) + ".npz")
            ys_dict = numpy.load(processed_training_data_dir + "data_ys_context_" + str(context_length) + ".npz")
            xs = xs_dict[xs_dict.keys()[0]].tolist()
            ys = ys_dict[ys_dict.keys()[0]].tolist()
            print("load done")
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

    # creates the neural network
    def create_network(self):
        # define input_data layer: shape tells us how the input data looks like. First element defines batch size and should be "None"
        self.net = tflearn.input_data(shape=[None, 2*context_length*len(self.string_to_number)]) # 2*context_length
        # add a deep layer with 32 nodes (and linear activation function?)
        self.net = tflearn.fully_connected(self.net, 32)
        # add a deep layer with softmax activation and as many nodes as one hot encoding length
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
        print("network created")
        self.model.fit(xs, ys, n_epoch=5, batch_size=128, show_metric=True)
        print("model trained")
        self.model.save(model_file)
        
    # gets a hole in the code, a.k.a prefix and suffix and predicts what has to be in the hole
    def query(self, prefix, suffix):
        # get prefix and suffix of desired length
        prefix, suffix = self.trim_to_length(prefix, suffix, context_length)
        # encode prefix and suffix and concatenate them for NN input
        x = self.one_hot_tokenlist(prefix) + self.one_hot_tokenlist(suffix)
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
    
    def trim_to_length(self, prefix:list, suffix:list, final_length:int):
        """
        receives a list of prefix and suffix tokens and trims them to the desired length, adding padding if necessary
        """

        if len(prefix) < final_length:
            #print("prefxix <")
            needed_padding = final_length - len(prefix)
            prefix = [{"type": "padding", "value": "padding"}] * needed_padding + prefix
        elif len(prefix) > final_length:
            #print("prefix >")
            prefix = prefix[-final_length:]

        if len(suffix) < final_length:
            #print("suffix <")
            needed_padding = final_length - len(suffix)
            suffix = suffix + [{"type": "padding", "value": "padding"}] * needed_padding
        elif len(suffix) > final_length:
            #print("suffix >")
            suffix = suffix[:final_length]
        #print("return from trim with prefix=")
        #print(prefix)
        #print("suffix=")
        #print(suffix)

        return (prefix, suffix)

    def one_hot_tokenlist(self, tokenlist):
        """
        takes a list of tokens, that are encoded as dicts and returns a list of concatenated one_hot_encodings of all tokens
        """
        one_hot_encodings = []
        for token in tokenlist:
            one_hot_encodings = one_hot_encodings + self.one_hot(self.token_to_string(token))

        return one_hot_encodings