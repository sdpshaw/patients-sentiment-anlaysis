import pandas as pd


class Utilities:
    def __int__(self, path):
        self.path = path

    def load_data(self):
        """
        Read the text file and return as dataframe
        """
        file = open(self.path, "r")
        s = file.readlines()
        input_text = []
        emotions = []
        for row in s:
            x = row.split(";")
            input_text.append(x[0])
            emotions.append(x[1][:-1])
        df = pd.DataFrame({"input_text": input_text, "emotions": emotions})
        return df
