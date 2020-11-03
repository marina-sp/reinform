import torch
from torch import nn

class PolicyStep(nn.Module):
    def __init__(self, option):
        super(PolicyStep, self).__init__()
        self.option = option
        self.lstm_cell = torch.nn.LSTMCell(input_size=self.option.action_embed_size,
                          hidden_size=self.option.state_embed_size)
    '''
    - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
    - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
    - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    '''
    def forward(self, prev_action, prev_state):
        output, new_state = self.lstm_cell(prev_action, prev_state)
        return output, (output, new_state)

class PolicyMlp(nn.Module):
    def __init__(self, option):
        super(PolicyMlp, self).__init__()
        self.option = option
        self.hidden_size = option.mlp_hidden_size
        self.mlp_l1 = nn.Linear(self.option.state_embed_size + self.option.action_embed_size,
                                self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(self.hidden_size, self.option.action_embed_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden)).unsqueeze(1)
        return output

class BertPolicy(nn.Module):
    def __init__(self, option):
        super().__init__()
        self.option = option
        self.to_state = nn.Linear(256, self.option.state_embed_size)

    def forward(self, seq_embs, *params):
        #if not self.option.use_cuda:
        #    seq_embs = seq_embs.cpu()
        # print(embs.shape, input.shape)
        #embs = embs.detach()
        if self.option.bert_state_mode == "avg_token":
            new_state = torch.tanh(self.to_state(seq_embs[:, 1:-1, :].mean(1)))
        elif self.option.bert_state_mode == "avg_all":
            new_state = torch.tanh(self.to_state(seq_embs.mean(1)))
        elif self.option.bert_state_mode == "sep":
            new_state = torch.tanh(self.to_state(seq_embs[:, -1, :]))
        elif self.option.bert_state_mode == "mask":
            new_state = torch.tanh(self.to_state(seq_embs[:, 1, :]))
        return new_state, torch.tensor([0.0])
