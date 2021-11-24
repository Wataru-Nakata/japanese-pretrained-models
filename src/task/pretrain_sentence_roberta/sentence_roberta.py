from transformers import RobertaModel, RobertaConfig
import torch.nn as nn


class SentenceRoberta(nn.Module):
    def __init__(self,input_dim:int,roberta_config:RobertaConfig) -> None:
        super().__init__()
        self.roberta = RobertaModel(roberta_config)
        self.roberta.embeddings.word_embeddings = None
        if not input_dim == roberta_config.hidden_size:
            self.use_input_linear = True
            self.input_linear = nn.Linear(input_dim,roberta_config.hidden_size)
            self.output_linear = nn.Linear(roberta_config.hidden_size,input_dim)
        else:
            self.use_input_linear = False
    def forward(self,roberta_input):
        if self.use_input_linear:
            input_embeds = self.input_linear(roberta_input['input_embeds'])
        else:
            input_embeds = roberta_input['input_embeds']
        out = self.roberta(
            attention_mask= roberta_input['attn_masks'],
            position_ids = roberta_input['position_ids'],
            inputs_embeds = input_embeds
        ).last_hidden_state
        if self.use_input_linear:
            out = self.output_linear(out)
        return out


