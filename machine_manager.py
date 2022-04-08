import pandas as pd
import numpy as np

class machine_manager:

    def __init__(self):
        self.machines = []

    def add_machine(self, machine):
        self.machines.append(machine)

    # Gets the data of a specific character
    def get_character_data(self, machine, data):
        list_index = data.index[data['label'] == machine.character].tolist()
        PC_total = data.drop('label', axis=1).iloc[list_index]
        return PC_total

    # Trains the GMM on a set of data.
    def train_machine(self, data):
        for character in self.machines:
            char_list = self.get_character_data(character, data)
            character.train(char_list)

    # Predicts the character
    def predict_machine(self, test_data):
        predictions = {}
        for character in self.machines: # Each machine predicts the chance of it being part of it
            predictions[character.character] = character.predict(test_data)
          
        all_best_guess = []
        for i in range(len(test_data)):
            best_guess = [-100,""]
            for character in self.machines:
              if predictions[character.character][i] > best_guess[0]:
                best_guess[0] = predictions[character.character][i]
                best_guess[1] = character.character
            all_best_guess.append(best_guess[1])

        return all_best_guess