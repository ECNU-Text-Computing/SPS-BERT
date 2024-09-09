import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, base_dir):
        """Initializes the DataProcessor class by setting up the directory where results will be stored."""
        self.results_dir = base_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, filename):
        """Loads and preprocesses data from a CSV file."""
        df = pd.read_csv(filename)
        return df

    def save_data(self, df, filename):
        """Saves DataFrame to JSON format with all its content."""
        df.to_json(os.path.join(self.results_dir, filename), orient='records', lines=True)

    def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Splits the data into training, validation, and test sets."""
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
        return train_df, val_df, test_df

    def process_data(self, filename):
        """Main method to load data, split it, and save the split data into separate files."""
        df = self.load_data(filename)
        # self.save_data(df, 'test_data.json')
        train_df, val_df, test_df = self.split_data(df)
        self.save_data(train_df, 'train_data.json')
        self.save_data(val_df, 'val_data.json')
        self.save_data(test_df, 'test_data.json')
        print("Data processing and saving complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset for model training.')
    parser.add_argument('--data', required=True, help='CSV file containing data.')
    parser.add_argument('--data_name', required=True, help='CSV file containing data.')
    args = parser.parse_args()

    # Create a directory name based on the input data file name
    base_dir = f'data_clean/{os.path.splitext(os.path.basename(args.data_name))[0]}/'
    data_processor = DataProcessor(base_dir)
    data_processor.process_data(args.data)