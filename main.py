import pandas as pd
from pyparsing import line
import machine_manager
import gaussian_mixture
import os
import sys

# Gives the predictive text 
def prediction_text(character, prediction, whole_data, user_data, lines, col):
    
    # Gets the data for the wanted character
    list_index = whole_data.index[whole_data['label'] == character].tolist()
    prediction_specific_data = whole_data.drop('label', axis=1).iloc[list_index]

    return_text = []
    if prediction == character:
        return_text.append(lines[0].format(get_character_name(character)))
    else:
        if len(lines) > 2:
            if get_average(prediction_specific_data, col) > get_average(user_data, col) and col != -1:
                return_text.append(lines[1])
            elif col != -1:
                return_text.append(lines[2])
        else:
            return_text.append(lines[1].format(get_character_name(prediction),get_character_name(character)))
    print(return_text[0] + ';', end = '')
    return return_text

# Get the names of the characters, returns a better version if it exists
def get_character_name(character_selected):
    classes = {"dwarf_f":"Dwarf Fighter", "human_s":"Human Sorcerer", "human_f":"Human fighter", "elf_ran":"Elf Ranger","batmanc":"Batman"}
    if character_selected in classes:
        classes_name = classes[character_selected]
    else:
        classes_name = character_selected

    return classes_name

# Gets average of a row given some data
def get_average(data, col):
    return data[data.columns[[col]]].mean()[0]

def main(file_name, character_selected = "dwarf_f"):
    # Get the cols of the data
    os.system('python feature_extraction.py 22050 ' + file_name + ' user --formants --ZCR --harmonics --rate_of_speech --pitch_features --spectral_features --energy')

    # Different voice attributes
    cols = {"overall":[3,4,8,9,10,11,12,16], "pitch":[8,9,16], "harmonics":[4,16], "shimmer":[10, 11, 16], "spectral":[12, 13,16]}

    # Lines for different alterations
    lines = {
        "overall":["You're sounding like a true {0}! Well done!", "You're sounding more like a {0} than a {1}"],
        "pitch":["Your pitch is good!", "Your pitch is a little low, try increasing it!", "Your pitch is a little high, try lowing it!"],
        "harmonics":["Your harmonics are good!", "Your harmonics are a little low, try raising the tone of your voice!", "Your harmonics are a little high, try decreasing the tone of your voice!"],
        "shimmer":["Your range is good!", "Your range is a little low, try being a little more wild with your voice!", "Your range is a little high, try to reduce the variation in your voice!"],
        "spectral":["Your voice quality is good!", "Your voice quality is a little off, try altering it. ie: Harsher, breathier, creaky voice."]
    }

    # Gets the features and unique labels from overall data
    df = pd.read_csv("./features.csv",sep=',')
    character_catagories = df.label.unique()

    # Gets the users data
    try:
        to_predict = pd.read_csv("./userfeatures.csv",sep=',')
    except:
        print("Error: No userfeatures.csv file. Add a .wav file to the directory and try again.")
        return

    predictions = {}

    for feature in cols:
        feature_data = df[df.columns[cols[feature]]]
        feature_to_predict = to_predict[to_predict.columns[cols[feature][:-1]]]

        # Set the emotions
        character_machines = machine_manager.machine_manager()

        # Create a GMM for each emotion
        for character in character_catagories:
            character_machine = gaussian_mixture.GMM(character) 
            character_machines.add_machine(character_machine)

        # Train the machines
        character_machines.train_machine(feature_data)
        # Predict the machines
        predictions[feature] = character_machines.predict_machine(feature_to_predict)

    final_texts = []

    # gets the text for what to change
    for feature in cols:

        final_texts = final_texts + prediction_text(character_selected, predictions[feature][0],
        feature_data,
        feature_to_predict,
        lines[feature],
        0)
        
    #print(predictions)
    #print(final_texts)
    return(final_texts)

if __name__ == '__main__':
    file_name  = sys.argv[1]
    character = sys.argv[2]
    main(file_name, character)