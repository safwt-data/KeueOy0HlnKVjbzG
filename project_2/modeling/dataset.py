import pandas as pd
import logging


def load_data():
    logging.info("Loading dataset")
    df = pd.read_csv("Data/term-deposit-marketing-2020.csv")
    logging.info("Dataset loaded successfully")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
