import tflearn
import tensorflow as tf
import numpy
import os
import random

# determines how many tokens before and after are to be used during training
context_length = 3
processed_training_data_dir = "/home/kurse/kurs00020/mj97nyma/jobs/ASDL2018/training_data/processed_training_data/"   # Path for Cluster
#processed_training_data_dir = "./../training_data/processed_training_data/"


class Code_Completion_Context_CNN:

    # converts a token, that is a dict into a string
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
        # randomly choose configuration of the network
        self.layers_count = random.randrange(2,4)
        self.layers_filter_count = [random.choice([16,32,64,96]) for i in range(self.layers_count)]
        self.layers_filter_size = [random.choice([8,16,32,64,len(self.string_to_number),2*len(self.string_to_number)]) for i in range(self.layers_count)]
        self.layers_maxpool_kernel_size = [random.choice([2,3,4]) for i in range(self.layers_count)]
        self.dropout = random.choice([True,True,False])
        if self.dropout:
            self.fully_connected_nodes = random.choice([32,64,96])
            self.droput_thresh = random.choice([0.25, 0.4, 0.5])
        self.epochs = random.choice([3,4])

        # Print Configuration to console
        print("Initializing Model with layers_count=%d, each with:"%self.layers_count)
        for layer in zip(self.layers_filter_count, self.layers_filter_size, self.layers_maxpool_kernel_size):
            print("filter_count=%d, filter_size=%d, maxpool_kernel_size=%d"%(layer[0],layer[1],layer[2]))
        if not self.dropout:
            print("No dropout Layer")
        else:
            print("Extra fully connected Layer with %d nodes and dropout threshold %1.2f"%(self.fully_connected_nodes, self.droput_thresh))
        print("Number of Epochs: %d"%self.epochs)


        # define input_data layer: shape tells us how the input data looks like. First element defines batch size and should be "None"
        self.net = tflearn.input_data(shape=[None, context_length*2*len(self.string_to_number), 1])
        # add Convolutional layers
        for i in range(self.layers_count):
            self.net = tflearn.layers.conv.conv_1d(self.net, self.layers_filter_count[i], self.layers_filter_size[i], activation='relu')     # 16 filters of size len(self.string_to_number)
            self.net = tflearn.layers.conv.max_pool_1d(self.net, self.layers_maxpool_kernel_size[i])

        if self.dropout:
            self.net = tflearn.fully_connected(self.net, self.fully_connected_nodes, activation='relu')
            self.net = tflearn.layers.core.dropout(self.net, self.droput_thresh)

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

        # reshape data to fit CNN
        xs = numpy.array(xs).reshape([-1,context_length*2*len(self.string_to_number),1]).tolist()

        self.create_network()
        self.model.fit(xs, ys, n_epoch=self.epochs, batch_size=512, show_metric=True)
        # self.model.save(model_file) # save file only if accuracy improved
        
    # gets a hole in the code, a.k.a prefix and suffix and predicts what has to be in the hole
    def query(self, prefix, suffix):
        # get prefix and suffix of desired length
        prefix, suffix = self.trim_to_length(prefix, suffix, context_length)
        # encode prefix and suffix and concatenate them for NN input
        x = self.one_hot_tokenlist(prefix) + self.one_hot_tokenlist(suffix)

        # reshape data to fit CNN
        x = numpy.array(x).reshape([-1, context_length*2*len(self.string_to_number),1]).tolist()

        y = self.model.predict(x) # müssen die klammer weg? Nämlich fehler!
        predicted_seq = y[0]
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