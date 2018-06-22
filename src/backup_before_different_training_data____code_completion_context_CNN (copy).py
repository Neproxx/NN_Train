import tflearn
import numpy
import os

# determines how many tokens before and after are to be used during training
context_length = 3
processed_training_data_dir = "./../training_data/processed_training_data/"
max_hole_size = 3

class Code_Completion_Context_CNN:

    # covnerts a token, that is a dict into a string
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    # converts a string, that represents a token into a dict
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
        all_token_strings.add("noToken-@@-noToken") # add a token to signal, that this represents no token (for prediction later on)
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
        self.net = tflearn.input_data(shape=[None, context_length*2*len(self.string_to_number), 1])
        # add Convolutional layers
        self.net = tflearn.layers.conv.conv_1d(self.net, 16, int(2*len(self.string_to_number)), activation='relu')     # 16 filters of size len(self.string_to_number)
        self.net = tflearn.layers.conv.max_pool_1d(self.net, 2)
        #self.net = tflearn.layers.conv.conv_1d(self.net, 16, int(2*len(self.string_to_number)), activation='relu')
        #self.net = tflearn.layers.conv.max_pool_1d(self.net, 2)         # TODO: Try out Drop Out Layers
        self.net = tflearn.fully_connected(self.net, max_hole_size*len(self.string_to_number), activation='softmax')
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
        self.model.fit(xs, ys, n_epoch=1, batch_size=512, show_metric=True)
        self.model.save(model_file)
        
    # gets a hole in the code, a.k.a prefix and suffix and predicts what has to be in the hole
    def query(self, prefix, suffix):
        # get prefix and suffix of desired length
        prefix, suffix = self.trim_to_length(prefix, suffix, context_length)
        # encode prefix and suffix and concatenate them for NN input
        x = self.one_hot_tokenlist(prefix) + self.one_hot_tokenlist(suffix)

        # reshape data to fit CNN
        x = numpy.array(x).reshape([-1, context_length*2*len(self.string_to_number),1]).tolist()

        y = self.model.predict(x)
        predicted_seq = y[0]    # hier muss ich die predicted sequence durch die max_hole_size splitten -> nutze dazu die prints unten drunter
        if type(predicted_seq) is numpy.ndarray:        # is this really necessary?
            predicted_seq = predicted_seq.tolist() 
        predicted = []
        predicted_count = int(len(predicted_seq)/len(self.string_to_number))
        print("predicted_count=%1.2f"%predicted_count)      # DEBUG print
        for i in range(predicted_count):
            predicted.append(predicted_seq[i*len(self.string_to_number):(i+1)*len(self.string_to_number)])
        highest_scores = [pred.index(max(pred)) for pred in predicted]
        #predicted_1 = predicted_seq[:len(self.string_to_number)]
        #predicted_2 = predicted_seq[len(self.string_to_number):2*len(self.string_to_number)]
        #predicted_3 = predicted_seq[2*len(self.string_to_number):]
        #best_numbers = [predicted_1.index(max(predicted_1)), predicted_2.index(max(predicted_2)), predicted_3.index(max(predicted_3))]
        #final_prediction = []
        #print(predicted_seq)

        
        #best_string = self.number_to_string[best_number]
        best_strings = [self.number_to_string[highest_score] for highest_score in highest_scores if not highest_score == one_hot("noToken-@@-noToken").index(max(one_hot("noToken-@@-noToken")))] # filter if "noToken" was predicted
        #best_token = self.string_to_token(best_string)
        best_tokens = [self.string_to_token(best_string) for best_string in best_strings]
        # return prediction as a list, containing the predicted token as a dict
        return best_tokens
    
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
        # wenn ich hier jetzt 3 von löchern predicte, dann muss ich irgendeinen threshold setzen für wann etwas tatsächlcih als predicted gilt und wann nicht
        # ich soltle mir am besten mal viele verschiedene sachen ausgeben lassen, was eignetlich so predicted wird

        """
        one_hot_encodings = []
        for token in tokenlist:
            one_hot_encodings = one_hot_encodings + self.one_hot(self.token_to_string(token))

        return one_hot_encodings