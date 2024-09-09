import os
import torch
import torch.nn as nn
from transformers import AutoModel

os.makedirs('sample_abs', exist_ok=True)  # Ensure the directory exists

class ONLYREF(nn.Module):
    """
    This class defines a PyTorch model that uses pre-trained SciBERT embeddings to process scientific
    abstracts and their associated references for regression tasks.
    """

    def __init__(self, dropout_rate):
        """
        Initializes the model components including SciBERT and dropout layers.

        Parameters:
        dropout_rate (float): The dropout rate used in the dropout layer to prevent overfitting.
        """
        super(ONLYREF, self).__init__()
        self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=8)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(self.bert.config.hidden_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, batch_x, batch_masks, token_type_ids, batch_refs_x):
        """
        Defines the forward pass of the model.

        Parameters:
        batch_x (Tensor): Input tensor for the abstracts.
        batch_masks (Tensor): Attention masks for the abstracts.
        token_type_ids (Tensor): Token type IDs for the abstracts (used by BERT to differentiate between segments).
        batch_refs_x (Tensor): Input tensor for the references.
        batch_index (int): Index of the current batch (unused in computation but could be useful for logging).

        Returns:
        Tensor: The regression output of the model.
        """
        abstract_embeddings = self.bert(batch_x, attention_mask=batch_masks).last_hidden_state
        abstract_embeddings = abstract_embeddings.mean(dim=1)
        # print("Abstract Embeddings -- Min:", abstract_embeddings.min().item(), "Max:", abstract_embeddings.max().item())

        batch_size, num_refs, seq_len = batch_refs_x.size()
        refs_embeddings = self.bert(batch_refs_x.view(-1, seq_len)).last_hidden_state
        refs_embeddings = refs_embeddings.view(batch_size, num_refs, seq_len, -1).mean(dim=2)
        # print("References Embeddings -- Min:", refs_embeddings.min().item(), "Max:", refs_embeddings.max().item())

        abstract_embeddings_a = abstract_embeddings.unsqueeze(0)
        refs_embeddings_a = refs_embeddings.permute(1, 0, 2)
        _, attn_weights = self.attention(abstract_embeddings_a, refs_embeddings_a, refs_embeddings_a)
        attn_weights = attn_weights.squeeze(1)
        refs_embeddings = (attn_weights.unsqueeze(2) * refs_embeddings).sum(dim=1)

        combined_embeddings = torch.cat([abstract_embeddings, refs_embeddings], dim=1)

        output1 = self.fc1(combined_embeddings)
        output = self.tanh(output1)
        output = self.fc2(output)

        return output
