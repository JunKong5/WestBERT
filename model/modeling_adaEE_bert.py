import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertLayer, BertPreTrainedModel


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, input_dim).cuda()

        self.pooling_linear = nn.Linear(input_dim, 1).cuda()

    def forward(self, x):
        self.pre_pooling_linear(x)
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)


class ATTGTeaKDLoss(nn.Module):
    def __init__(self, nBlocks, gamma, T, num_labels):
        super(ATTGTeaKDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.atten = SelfAttention(num_labels)
        self.nBlocks = nBlocks
        self.gamma = gamma
        self.T = T
        self.num_labels = num_labels


    def forward(self, outputs, eachlayer_logits_all, targets, soft_targets):

        T = self.T
        multi_celosses = []
        distill_losses = []
        eachlayer_logits_all = eachlayer_logits_all[:-1]
        eachlayer_logits_all.append(outputs)

        eachlayer_logits_alls = torch.tensor([item.cpu().detach().numpy() for item in eachlayer_logits_all]).cuda()
        eachlayer_logits_alls = eachlayer_logits_alls.transpose(0, 1)

        tealogits = self.atten(eachlayer_logits_alls)


        if self.num_labels == 1:
            tea_loss = self.mse_loss(tealogits.view(-1), targets.view(-1))
        else:
            tea_loss = self.ce_loss(tealogits.view(-1, self.num_labels), targets.view(-1))



        for i in range(self.nBlocks):
            if self.num_labels == 1:
                _mse = (1. - self.gamma) * self.mse_loss(eachlayer_logits_all[i].view(-1), targets.view(-1))
                _kld = self.kld_loss(self.log_softmax(eachlayer_logits_all[i].view(-1) / T),
                                     self.softmax(soft_targets.view(-1) / T)) * self.gamma * T * T
                multi_celosses.append(_mse)
                distill_losses.append(_kld)
            else:
                _ce = (1. - self.gamma) * self.ce_loss(eachlayer_logits_all[i].view(-1, self.num_labels), targets.view(-1))
                _kld = self.kld_loss(self.log_softmax(eachlayer_logits_all[i].view(-1, self.num_labels) / T),
                                     self.softmax(tealogits.view(-1, self.num_labels) / T)) * self.gamma * T * T

                multi_celosses.append(_ce)
                distill_losses.append(_kld)


        m_loss = sum(multi_celosses)
        d_loss = sum(distill_losses)
        tealoss = tea_loss*self.gamma

        loss = d_loss + m_loss + tealoss

        return (loss, d_loss, m_loss, tea_loss)



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.EElayer = nn.ModuleList([BertEElayer (config) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

    def set_early_exit_entropy(self, x):
        print(x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_early_exit_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for EElayer  in self.EElayer:
            for name, param in EElayer.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_EElayer_exits = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            EElayer_exit = self.EElayer[i](current_outputs)



            if not self.training:
                EElayer_logits = EElayer_exit[0]
                EElayer_feature = EElayer_exit[1]
                EElayer_entropy = entropy(EElayer_logits)
                EElayer_exit = EElayer_exit + (EElayer_entropy,)
                all_EElayer_exits = all_EElayer_exits + (EElayer_exit,)

                if EElayer_entropy < self.early_exit_entropy[i]:

                    new_output = (EElayer_logits,) + (EElayer_feature,) + current_outputs[1:] + (all_EElayer_exits ,)
                    raise EElayerException(new_output, i+1)
            else:
                all_EElayer_exits  = all_EElayer_exits  + (EElayer_exit,)


        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = outputs + (all_EElayer_exits ,)
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def init_early_exit_pooler(self):
        self.encoder.init_early_exit_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs


class EElayerException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!


class BertEElayer (nn.Module):

    def __init__(self, config):
        super(BertEElayer, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_layer=-1,kd_loss_type='kd', gamma=0.1, temper=3.0):

        exit_layer = self.num_layers
        try:
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)

            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except EElayerException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        eachlayer_logits_all = []
        eachlayer_feature_all = []
        if not self.training:
            original_entropy = entropy(logits)
            eachlayer_entropy = []


            
        if labels is not None:

            # work with eachlayer exits

            for EElayer_exit in outputs[-1]:
                EElayer_logits = EElayer_exit[0]
                EElayer_features = EElayer_exit[1]
                eachlayer_logits_all.append(EElayer_logits)
                eachlayer_feature_all.append(EElayer_features)
                if not self.training:
                    eachlayer_entropy.append(EElayer_exit[2])

            if kd_loss_type == 'kd':
                loss_fct = ATTGTeaKDLoss(len(outputs[-1]), gamma, temper, self.num_labels)
                soft_labels = logits.detach()
                loss_kd = loss_fct(logits, eachlayer_logits_all, labels, soft_labels)
                outputs = (loss_kd,) + outputs
            else:
                if self.num_labels == 1:
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        if not self.training:
            outputs = outputs + ((original_entropy, eachlayer_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (eachlayer_logits_all[output_layer],) +\
                          outputs[2:]

        return outputs
