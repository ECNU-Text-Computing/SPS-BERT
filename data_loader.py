import os
import torch
import random
import pandas as pd
from transformers import AutoTokenizer

class DataLoader:
    """
    DataLoader class to handle the loading and preprocessing of text data for model training.
    This includes tokenization, padding, and batching of data.
    """
    def __init__(self, model_name="allenai/scibert_scivocab_uncased"):
        """
        Initializes the DataLoader with a specific tokenizer and sets the parameters for padding.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_size = 512  # Max length of tokenized sequences

    def load_data(self, filepath):
        """
        Loads a DataFrame from a JSON file and returns it.
        """
        df = pd.read_json(filepath, orient='records', lines=True)
        return df

    def data_generator(self, data_path, batch_size, label_select, shuffle=False):
        """
        Generates data batches for training from a single file containing all data.
        """
        df = self.load_data(data_path)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(df), batch_size):
            batch_data = self._process_batch(df['abstract'].iloc[i:i+batch_size])
            batch_refs = self._process_references(df['ref_titles'].iloc[i:i+batch_size])
            batch_y = torch.tensor(df[label_select].iloc[i:i+batch_size].tolist(), dtype=torch.float)
            yield (*batch_data, batch_refs, batch_y)

    def _process_batch(self, texts):
        """
        Helper function to process a batch of texts into model inputs.
        """
        batch_token_ids, batch_mask, batch_token_type_ids = [], [], []
        for text in texts:
            tokens = self.tokenizer.tokenize(text.strip())[:self.pad_size]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding_length = self.pad_size - len(token_ids)
            batch_token_ids.append(token_ids + [0] * padding_length)
            batch_mask.append([1] * len(token_ids) + [0] * padding_length)
            batch_token_type_ids.append([0] * self.pad_size)

        return torch.tensor(batch_token_ids, dtype=torch.long), \
               torch.tensor(batch_mask, dtype=torch.long), \
               torch.tensor(batch_token_type_ids, dtype=torch.long)

    def _process_references(self, references_list, ref_size=20, max_refs=20):
        """
        Helper function to process reference texts for each abstract.
        Adjusts all sequences to a fixed length `ref_size` for consistent tensor conversion,
        and ensures each batch contains the same number of references `max_refs`, each of fixed length `ref_size`.

        Parameters:
        - references_list (list of list of str): List of reference lists, where each inner list is a set of references for one sample in the batch.
        - ref_size (int): Fixed length for each reference.
        - max_refs (int): Fixed number of references per batch item.

        Returns:
        - torch.Tensor: Tensor of shape (batch_size, max_refs, ref_size) containing token IDs for references.
        """
        batch_refs_token_ids = []
        for refs in references_list:
            # Process each reference, truncate or pad to `ref_size`, and ensure not more than `max_refs` are processed
            refs_token_ids = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ref.strip())[:ref_size])
                for ref in refs[:max_refs]
            ]
            # Pad each reference to `ref_size`
            refs_token_ids = [ids + [0] * (ref_size - len(ids)) for ids in refs_token_ids]
            # If there are fewer than `max_refs`, pad the remaining with zero vectors
            refs_token_ids += [[0] * ref_size] * (max_refs - len(refs_token_ids))
            batch_refs_token_ids.append(refs_token_ids)

        # Convert to a tensor of shape (batch_size, max_refs, ref_size)
        return torch.tensor(batch_refs_token_ids, dtype=torch.long)