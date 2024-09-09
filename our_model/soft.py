import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    """
    SoftEmbedding extends a pre-trained embedding layer with additional, learnable embeddings.

    Parameters:
    - pretrained_embedding (nn.Embedding): The pre-existing, pre-trained embedding layer.
    - n_tokens (int): Number of additional embedding vectors to learn.
    - random_range (float): Range for random initialization of new embeddings.
    - initialize_from_vocab (bool): If True, initialize new embeddings from the existing vocabulary.
    - tag (str): Optional tag for the embeddings (unused in functionality but can be useful for identification).

    Attributes:
    - learned_embedding (Parameter): New embeddings that are learned.
    """
    def __init__(self,
                 pretrained_embedding: nn.Embedding,
                 n_tokens: int = 6,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True,
                 tag: str = ''):
        super(SoftEmbedding, self).__init__()
        self.n_tokens = n_tokens
        self.tag = tag
        self.pretrained_embedding = pretrained_embedding
        # Initialize the learned embeddings
        self.learned_embedding = nn.Parameter(self.initialize_embedding(n_tokens,
                                                                        random_range,
                                                                        initialize_from_vocab))

    def initialize_embedding(self,
                             n_tokens: int,
                             random_range: float,
                             initialize_from_vocab: bool):
        """
        Initializes additional embeddings either by copying from the pre-trained weights or
        by random initialization.

        Returns:
        Tensor: The initialized embedding matrix for the new tokens.
        """
        if initialize_from_vocab:
            # Initialize from the first `n_tokens` entries of the pre-trained embeddings
            return self.pretrained_embedding.weight[:n_tokens].clone().detach()
        else:
            # Randomly initialize the embeddings within a specified range
            return torch.FloatTensor(n_tokens, self.pretrained_embedding.embedding_dim).uniform_(-random_range, random_range)

    def forward(self, input_ids):
        """
        Extends the input embedding with additional learned embeddings.

        Parameters:
        - input_ids (Tensor): Indices to embed via the pre-trained embeddings.

        Returns:
        Tensor: Combined embeddings with pre-trained and additional learned embeddings appended.
        """
        input_embedding = self.pretrained_embedding(input_ids)
        # Repeat the learned embeddings for each example in the batch to match the batch size
        learned_embedding = self.learned_embedding.unsqueeze(0).repeat(input_embedding.size(0), 1, 1)
        # Concatenate the original and new embeddings along the sequence dimension
        combined_embedding = torch.cat([learned_embedding, input_embedding], dim=1)

        return combined_embedding