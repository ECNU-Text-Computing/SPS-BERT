import torch
import torch.nn as nn
from transformers import AutoModel
from soft import SoftEmbedding

class SPSBERT(nn.Module):
    """
    SPSBERT integrates SciBERT with custom soft embeddings for abstracts and references and applies attention
    to merge the embeddings from both sources, followed by a feed-forward network with batch normalization 
    and ReLU activation for improved prediction tasks. Outputs are scaled to -1 to 1 using the tanh activation function.
    """
    def __init__(self, dropout_rate, **kwargs):
        super(SPSBERT, self).__init__()
        model_name = "allenai/scibert_scivocab_uncased"
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        self.bert_abstract = SoftEmbedding(self.bert.get_input_embeddings(),
                                           n_tokens=5,
                                           initialize_from_vocab=True,
                                           tag='abstract')
        self.bert_reference = SoftEmbedding(self.bert.get_input_embeddings(),
                                            n_tokens=5,
                                            initialize_from_vocab=True,
                                            tag='reference')

        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=8)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(self.bert.config.hidden_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, batch_x, batch_masks, token_type_ids, batch_refs_x):
        self.bert.set_input_embeddings(self.bert_abstract)
        abstract_embeddings = self.bert(batch_x, attention_mask=batch_masks).last_hidden_state
        abstract_embeddings = abstract_embeddings.mean(dim=1)
        # print("Abstract Embeddings -- Min:", abstract_embeddings.min().item(), "Max:", abstract_embeddings.max().item())

        self.bert.set_input_embeddings(self.bert_reference)
        batch_size, num_refs, seq_len = batch_refs_x.size()
        refs_embeddings = self.bert(batch_refs_x.view(-1, seq_len)).last_hidden_state
        refs_embeddings = refs_embeddings.view(batch_size, num_refs, seq_len, -1).mean(dim=2)
        # print("References Embeddings -- Min:", refs_embeddings.min().item(), "Max:", refs_embeddings.max().item())

        abstract_embeddings_a = abstract_embeddings.unsqueeze(0)
        refs_embeddings_a = refs_embeddings.permute(1, 0, 2)
        _, attn_weights = self.attention(abstract_embeddings_a, refs_embeddings_a, refs_embeddings_a)
        attn_weights = attn_weights.squeeze(1)
        refs_embeddings = (attn_weights.unsqueeze(2) * refs_embeddings).sum(dim=1)
        # combined_embeddings = torch.cat([abstract_embeddings, refs_embeddings], dim=1)
        combined_embeddings = abstract_embeddings + refs_embeddings
        output1= self.fc1(combined_embeddings)
        output = self.tanh(output1)
        output = self.fc2(output)

        return output