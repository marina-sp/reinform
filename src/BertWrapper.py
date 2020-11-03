import torch
from transformers import BertForMaskedLM, BertConfig
import copy

class BertWrapper():

    def __init__(self, option, data_loader):
        self.option = option
        self.data_loader = data_loader
        self.path_scoring_model = BertForMaskedLM.from_pretrained(option.bert_path)
        self.device = torch.device('cuda' if self.option.use_cuda else 'cpu')

    def make_trainable(self):
        self.train(self.option.train_layers == [])
        for name, par in self.path_scoring_model.named_parameters():
            if any([f'layer.{id}' in name for id in
                    self.option.train_layers]) or 'cls' in name and self.option.train_layers != []:
                par.requires_grad_(True)
                print(f"Layer {name} - activate training")
            else:
                par.requires_grad_(False)
                print(name)
        print(self.path_scoring_model)

    def embed_path(self, sequences, use_labels=False, test_mode=False):
        '''
        Prepare given sequence and make prediction scores.
        :param sequences: full paths e, r, e ..., r, e
        :param use_labels: whether to calculate loss using the correct label from the path
        :return: (prediction loss), predicted scores
        '''
        drop_tokens = self.option.token_droprate != 0.0

        if not test_mode and drop_tokens:
            dropout_mask = torch.bernoulli(torch.ones_like(sequences) * self.option.token_droprate).long()
            sequences[dropout_mask == 1] = self.data_loader.kg.unk_token_id
            # print((prev_action_embedding == self.data_loader.kg.unk_token_id).float().mean())

        cls = torch.ones(sequences.shape[0], 1).type(torch.int64) * self.data_loader.kg.cls_token_id
        sep = torch.ones(sequences.shape[0], 1).type(torch.int64) * self.data_loader.kg.sep_token_id
        inputs = copy.deepcopy(sequences).cpu()
        inputs[:, 0] = self.data_loader.kg.mask_token_id
        inputs = torch.cat((cls, inputs, sep), dim=-1).type(torch.int64).to(self.device)

        word_embeddings = self.path_scoring_model.bert.embeddings.word_embeddings(inputs.type(torch.int64))

        if test_mode and drop_tokens:
            word_embeddings *= 1 - self.option.token_droprate

        if use_labels:
            labels = torch.ones_like(inputs) * -1
            labels[:, 1] = sequences[:, 0]
            labels = labels.to(self.device)

            loss, probs, _ = self.path_scoring_model(inputs_embeds=word_embeddings,
                                                      masked_lm_labels=labels.type(torch.int64))
            return loss, probs[:,1,:]
        else:
            _, embs = self.path_scoring_model(inputs_embeds=word_embeddings)  # probs, embs
            return embs.cpu()
