'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radii of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights. 
'''

# Written and tested in Python 3.10.4

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import glorot_normal
import os
import time
import argparse
import json
import shutil

RANDOM_SEED = 42

def load_data(data_path:str, percent_test=0.2):
    # creates file paths and loads the data
    path_x = data_path+"_val.csv"
    path_y = data_path+".csv"
    data_x = np.genfromtxt(path_x,delimiter=',',dtype='float32')
    data_y = np.transpose(np.genfromtxt(path_y,delimiter=',',dtype='float32'))
    # normalizes the data (use these values to normalize any future data)
    mean, std = data_x.mean(), data_x.std()
    data_x = (data_x-mean)/std
    # splits the data into train/test groups
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=float(percent_test),random_state=RANDOM_SEED)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y,test_size=0.5,random_state=RANDOM_SEED)
    train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
    val_x = tf.convert_to_tensor(val_x, dtype=tf.float32)
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)
    val_y = tf.convert_to_tensor(val_y, dtype=tf.float32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.float32)
    return train_x, val_x, test_x, train_y, val_y, test_y, mean, std

def define_model(input_shape, output_shape, lr_rate, lr_decay, end_step):
    # metric_list provides the metrics tracked and saved
    metric_list = ['accuracy', 'MeanSquaredError']
    model = Sequential()
    initializer=glorot_normal(RANDOM_SEED)
    # add model layers
    model.add(Dense(256, input_shape=(input_shape[1],), activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(output_shape[1]))
    model.compile(optimizer=RMSprop(learning_rate=lr_rate), loss=MeanAbsoluteError(), metrics=metric_list)
    print(model.summary())
    return model

def evaluate_model(train_x, val_x, train_y, val_y, batch_size, num_epochs, lr_rate, lr_decay, percent_val):
    # Assume that input and output are 1D arrays, if not, adjust as needed
    input_shape, output_shape = train_x.shape, train_y.shape
    end_step = np.ceil(input_shape[0]*(1-percent_val))*num_epochs
    model = define_model(input_shape, output_shape, lr_rate, lr_decay, end_step)
    history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(val_x, val_y), verbose=1)
    return history, model

def plot_vs_epoch(name, values, output_folder):
    path = str(output_folder + '/figures')
    name =  name.capitalize()
    x = np.arange(len(values))
    save_path = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.plot(x,values)
    plt.title(name + " vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.savefig(save_path)
    plt.show()
    return

def save_json(data, name, output_folder):
    file_name = str(output_folder + '/' + name + '.json')
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print("saved " + str(name) + '.json at ' + str(output_folder))
    return

def set_folder(output_folder):
    print("You can either delete the current results at " + str(output_folder) + 'or you can create new folder to save the data')
    choice = input("Type \'delete\' to delete or \'new\' to create a new folder: ")
    while True:
        if choice == 'delete':
            shutil.rmtree(output_folder)
            break
        elif choice == 'new':
            new_path = input("Please type in the new folder name: ")
            output_folder = new_path
            if os.path.exists(output_folder):
                print("path already exists, pick a new one")
                break
            else:
                break
    return output_folder

def main(data, reuse_weights, output_folder, weight_name_save, weight_name_load, batch_size, num_epochs, lr_rate, lr_decay, num_layers, n_hidden, percent_val, patienceLimit, compare, sample_val, spect_to_sample, matchSpectrum, match_test_file, designSpectrum, design_test_file):
    # shutil.rmtree(output_folder) 
    if os.path.exists(output_folder):
        output_folder = set_folder(output_folder)
    train_x, val_x, test_x, train_y, val_y, test_y, mean, std = load_data(data, percent_val)
    print("Beginning training")
    history, model = evaluate_model(train_x, val_x, train_y, val_y, batch_size, num_epochs, lr_rate, lr_decay, percent_val)
    print("Completed training")
    # The folder model saves the data for the model
    print(model.summary())
    model_path = os.path.join(output_folder, 'model')
    model.save(model_path)
    histories = history.history
    stats = {'mean':float(mean), 'std':float(std)}
    scores = model.evaluate(x=test_x, y=test_y, return_dict=True)
    print(scores)
    save_json(histories, 'histories', output_folder)
    save_json(stats, 'stats', output_folder)
    save_json(scores, 'scores', output_folder)
    for key in histories:
        plot_vs_epoch(key, histories[key], output_folder)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/8_layer_tio2') # Where the data file is. Note: This assumes a file of _val.csv and .csv 
    parser.add_argument("--reuse_weights",type=str,default='False') # Whether to load the weights or not. Note this just needs to be set to true, then the output folder directed to the same location. 
    parser.add_argument("--output_folder",type=str,default='test_results') #Where to output the results to. Note: No / at the end. 
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes in front of w_1.txt. This would be used in saving the weights. In most cases, just leave this as is, it will naturally take care of it. 
    parser.add_argument("--weight_name_save",type=str,default="") #Similar to above, but for saving now. 
    parser.add_argument("--batch_size",type=int,default=64) # Batch Size
    parser.add_argument("--num_epochs",type=int,default=1000) #Max number of epochs to consider at maximum, if patience condition is not met. 
    parser.add_argument("--lr_rate",type=float,default=.001) # Learning Rate. 
    parser.add_argument("--lr_decay",type=float,default=.999) # Learning rate decay. It decays by this factor every epoch.
    parser.add_argument("--num_layers",default=4) # Number of layers in the network. 
    parser.add_argument("--n_hidden",default=225) # Number of neurons per layer. Fully connected layers. 
    parser.add_argument("--percent_val",default=.2) # Amount of the data to split for validation/test. The validation/test are both split equally. 
    parser.add_argument("--patience",type=int,default=10) # Patience for stopping. If validation loss has not decreased in this many steps, it will stop the training. 
    parser.add_argument("--compare",default='False') # Whether it should output the comparison or not. 
    parser.add_argument("--sample_val",default='True') # Wether it should sample from validation or not, for the purposes of graphing. 
    parser.add_argument("--spect_to_sample",type=int,default=300) # Zero Indexing for this. Position in the data file to sample from (note it will take from validation)
    parser.add_argument("--matchSpectrum",default='False') # If it should match an already existing spectrum file. 
    parser.add_argument("--match_test_file",default='results/2_layer_tio2/test_47.5_45.3') # Location of the file with the spectrum in it. 
    parser.add_argument("--designSpectrum",default='False') # If it should 
    parser.add_argument("--design_test_file",default='data/test_gen_spect.csv') # This is a file that should contain 0's and 1's where it should maximize and not maximize. 

    args = parser.parse_args()
    dict = vars(args)
    print(dict)

    for key,value in dict.items():
        if (dict[key]=="False"):
            dict[key] = False
        elif dict[key]=="True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    # print (dict)

    #Note that reuse MUST be set to true.
    if (dict['compare'] or dict['matchSpectrum'] or dict['designSpectrum']):
        if dict['reuse_weights'] != True:
            print("Reuse weights must be set true for comparison, matching, or designing. Setting it to true....")
            time.sleep(1)
        dict['reuse_weights'] = True
        
    kwargs = {  
            'data':dict['data'],
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'batch_size':dict['batch_size'],
            'num_epochs':dict['num_epochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':int(dict['num_layers']),
            'n_hidden':int(dict['n_hidden']),
            'percent_val':dict['percent_val'],
            'patienceLimit':dict['patience'],
            'compare':dict['compare'],
            'sample_val':dict['sample_val'],
            'spect_to_sample':dict['spect_to_sample'],
            'matchSpectrum':dict['matchSpectrum'],
            'match_test_file':dict['match_test_file'],
            'designSpectrum':dict['designSpectrum'],
            'design_test_file':dict['design_test_file']
            }
    main(**kwargs)