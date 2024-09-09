from transformers import AutoModel
import torch.nn as nn

class SCIBERT(nn.Module):
    """
    A PyTorch module for using SciBERT for regression tasks, specifically designed for scientific text.
    Includes a dropout layer and a linear regressor to predict continuous variables.

    Attributes:
        bert (AutoModel): The pre-trained SciBERT model loaded from Hugging Face.
        dropout (Dropout): Dropout layer to prevent overfitting.
        regressor (Linear): A linear layer for regression.
        metrics_num (int): Optional attribute to track the number of metrics, with a default value of 4.
    """
    def __init__(self, dropout_rate, **kwargs):
        """
        Initializes the SCIBERT model with a specified dropout rate and optional metrics tracking.

        Parameters:
            dropout_rate (float): The dropout probability to be used in the dropout layer.
            **kwargs (dict): Additional keyword arguments, specifically allows 'metrics_num' to specify the number of metrics.
        """
        super(SCIBERT, self).__init__()
        model_name = "allenai/scibert_scivocab_uncased"
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

        # Initialize metrics_num with a default or provided value.
        self.metrics_num = kwargs.get('metrics_num', 4)

        # Freeze the BERT parameters to prevent them from being updated during training.
        for param in self.bert.parameters():
            param.requires_grad = False
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids, batch_refs_x=None):
        """
        Defines the forward pass for the model.

        Parameters:
            input_ids (Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (Tensor): Segment token indices to indicate first and second portions of the inputs.
            batch_refs_x (Tensor, optional): Additional tensor for reference data, not used in this model.

        Returns:
            Tensor: The output of the regression layer.
        """
        # print(input_ids.size())
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.regressor(pooled_output)
        output = self.tanh(output)
        return output