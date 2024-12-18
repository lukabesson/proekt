import pandas as pd


def get_text():
    data = pd.read_csv("/turism.csv")

    text = ""

    for i in data['description']:
        text += i + "\n\n"

    return text



