from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from soft import SoftEmbedding

class ONLYPRO(nn.Module):
    """
    A custom PyTorch module for a model that integrates SciBERT from Hugging Face with a soft embedding layer
    and a linear classifier for regression tasks.

    The model can handle additional tokens that are softly integrated into the SciBERT embeddings.

    Attributes:
        bert (AutoModel): The pre-trained SciBERT model loaded from Hugging Face.
        dropout (Dropout): Dropout layer to prevent overfitting.
        classifier (Linear): Linear regression layer to predict a continuous variable.
        metrics_num (int): Optional attribute to track the number of metrics, default is 4.
    """
    def __init__(self, dropout_rate, **kwargs):
        super(ONLYPRO, self).__init__()
        model_name = "allenai/scibert_scivocab_uncased"
        self.bert = AutoModel.from_pretrained(model_name)

        # Enhance BERT with SoftEmbedding for additional tokens
        original_embeddings = self.bert.get_input_embeddings()
        self.bert.embeddings = SoftEmbedding(original_embeddings, n_tokens=5, initialize_from_vocab=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # Configurable number of metrics with a default of 4 if not specified
        self.metrics_num = kwargs.get('metrics_num', 4)

        # Optionally freeze all parameters in the BERT model to prevent updating during training
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, batch_refs_x=None):
        """
        Forward pass through the model.

        Parameters:
            input_ids (Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            batch_refs_x (Tensor, optional): Additional tensor for reference data, not used in this model.

        Returns:
            Tensor: The output of the regression layer.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Assuming outputs[1] is the pooled output from the model
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)
        return output
