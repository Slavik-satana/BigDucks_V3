from torch import nn
from transformers import AutoModel


class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes, dropout_prob=0.3):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, 368)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(368, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        pooled_output = outputs[
            1
        ]  # Предполагается, что outputs = (last_hidden_state, pooled_output)
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
