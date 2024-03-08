from transformers import MT5EncoderModel, MT5Config
from transformers.models.mt5 import MT5Stack
from transformers.modeling_outputs import BaseModelOutput
import torch
from typing import Dict, List, Optional, Tuple
import copy


class StructuredGlossTransformer(torch.nn.Module):
    """Seq2seq transformer variant that uses structured prediction for certain output tokens."""

    def __init__(self,
                 num_glosses: int,
                 feature_lengths: List[int],
                 feature_map: List[List[int]],
                 decoder_pad_token_id: int,
                 decoder_max_length: int) -> None:
        """
        Initializes a seq2seq transformer that uses features for prediction rather than single tokens.
        The first row of features is special in that it is used first to select the class absolutely.
        Then, the remaining features are used to calculate cosine similarity to produce the final prediction.

        Args:
            num_glosses (int): The number of glosses.
            feature_lengths (List[int]): List of ints, where each int defines the number of options for each feature.
            feature_map (List[List[int]]): A feature map. Should be (num_glosses, num_features). Each sublist is the feature values for a particular gloss.
            decoder_pad_token_id (int): The pad token ID for the decoder.
        """
        assert len(feature_map) == num_glosses, "Feature map does not match number of glosses"
        assert len(feature_map[0]) == len(feature_lengths), "Feature map does not match number of features"

        super().__init__()

        self.num_glosses = num_glosses
        self.feature_map = feature_map
        self.decoder_pad_token_id = decoder_pad_token_id
        self.decoder_max_length = decoder_max_length

        # Since we use different vocabulary for encoder (text) and decoder (glosses), initialize separately
        # Use pretrained MT5 encoder for crosslingual transfer
        config = MT5Config.from_pretrained("google/mt5-small")
        self.encoder = MT5EncoderModel(config=copy.deepcopy(config))

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = decoder_config.num_decoder_layers
        # TODO: Do we want to include dimensions in the decoder inputs?
        self.decoder_embedding = torch.nn.Embedding(self.num_glosses, decoder_config.d_model)
        self.decoder = MT5Stack(decoder_config, self.decoder_embedding)

        feature_heads = []
        if feature_lengths is not None:
            # Use multiple heads for output
            for feature_length in feature_lengths:
                feature_heads.append(torch.nn.Linear(config.d_model, feature_length, bias=False))
        else:
            # Use a single, standard LM head
            feature_heads.append(torch.nn.Linear(config.d_model, num_glosses, bias=False))
        self.feature_heads = torch.nn.ModuleList(feature_heads)
        """Maps labels to a bundle of feaure heads that define options for the label"""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        features: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.FloatTensor]]:
        """
        Forward pass of the model.

        Args:
            input_ids (Optional[torch.LongTensor], optional): Input tensor of shape `(batch_size, seq_length)`.
            attention_mask (Optional[torch.FloatTensor], optional): Attention mask tensor of shape `(batch_size, seq_length)`.
            decoder_input_ids (Optional[torch.LongTensor], optional): Decoder input tensor of shape `(batch_size, seq_length)`. If omitted, will start new output sequence.
            features (Optional[torch.LongTensor], optional): Feature tensor of shape `(batch_size, num_features, seq_length)`. Used to compute loss for decomposed prediction.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing the loss tensor and a list of feature logits tensors `(num_features, batch_size, seq_length, feature_size)`.
        """
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]
        if features is not None:
            assert features.shape[1] == len(self.feature_heads)
            assert input_ids.shape[0] == features.shape[0]

        encoder_outputs: Tuple[torch.FloatTensor] = self.encoder.forward(input_ids=input_ids,
                                                                         attention_mask=attention_mask,
                                                                         return_dict=False)

        if decoder_input_ids is not None:
            # Shift labels right as input to decoder
            # If we hit the max decoder length, we can truncate
            if features is None:
                new_seq_length = min(decoder_input_ids.shape[-1] + 1, self.decoder_max_length)
            else:
                new_seq_length = decoder_input_ids.shape[-1]
            shifted_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape[:-1] + (new_seq_length,))
            shifted_decoder_input_ids[..., 1:] = decoder_input_ids[..., :new_seq_length-1].clone()
            shifted_decoder_input_ids[..., 0] = self.decoder_pad_token_id
            shifted_decoder_input_ids.masked_fill_(shifted_decoder_input_ids == -100, self.decoder_pad_token_id)
        else:
            # Create a new tensor of size (batch_size, 1) that contains the start token
            shifted_decoder_input_ids = torch.full(
                (input_ids.shape[0], 1), self.decoder_pad_token_id, dtype=torch.long).to(input_ids.device)

        # (batch size, sequence_length, hidden_state_size)
        decoder_outputs: Tuple[torch.FloatTensor] = self.decoder.forward(input_ids=shifted_decoder_input_ids,
                                                                         encoder_hidden_states=encoder_outputs[0],
                                                                         encoder_attention_mask=attention_mask,
                                                                         return_dict=False)
        loss = 0
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # For each feature, compute logits and sum loss
        all_feature_logits = []
        for feature_index, feature_head in enumerate(self.feature_heads):
            feature_logits = feature_head.forward(decoder_outputs[0])
            all_feature_logits.append(feature_logits)

            if features is not None:
                loss += loss_fct(feature_logits.view(-1, feature_logits.size(-1)),
                                 features[:, feature_index].view(-1))

        return (loss, all_feature_logits)

    def greedy_decode(self, feature_logits: List[torch.Tensor]):
        """Decodes a bundle of feature logits into gloss ID predictions. 
        The output should align with the input vocabulary of the decoder.

        Args:
            feature_logits (List[torch.Tensor]): List of feature logit tensors each of size `(batch_size, seq_length, feature_size)`
        """
        batch_size, seq_length, _ = feature_logits[0].shape
        log_softmax = torch.nn.LogSoftmax(dim=-1)

        # num_features, batch_size, seq_length, feature_size
        softmaxed_logits = [log_softmax(logits) for logits in feature_logits]

        primary_features = torch.argmax(feature_logits[0], -1)  # batch_size, seq_length

        decoded_ids = torch.empty(batch_size, seq_length, dtype=torch.long).to(feature_logits[0].device)

        for row_index in range(batch_size):
            for token_index in range(seq_length):
                primary_feature = primary_features[row_index, token_index].item()
                possible_label_ids = [index for index, value in enumerate(
                    self.feature_map) if value[0] == primary_feature]

                if len(possible_label_ids) == 1:
                    decoded_ids[row_index, token_index] = possible_label_ids[0]
                else:
                    # Otherwise, try each possible label, choose one with highest probability
                    most_probable_label = (None, None)
                    for possible_label_id in possible_label_ids:
                        possible_label_features = self.feature_map[possible_label_id]
                        label_prob = 0
                        # Sum up log probabilities for each true feature of the label
                        for feature_index, feature_value in enumerate(possible_label_features):
                            if feature_index == 0:
                                continue
                            label_prob += softmaxed_logits[feature_index - 1][row_index, token_index, feature_value]

                        if most_probable_label[0] is None or label_prob > most_probable_label[0]:
                            most_probable_label = (label_prob, possible_label_id)
                    decoded_ids[row_index, token_index] = most_probable_label[1]

                # If we see a PAD token, set the remaining tokens to PAD and stop
                if decoded_ids[row_index, token_index] == self.decoder_pad_token_id:
                    decoded_ids[row_index, token_index:] = self.decoder_pad_token_id
                    break

        return decoded_ids

    def generate(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor],
    ) -> torch.Tensor:
        """Autoregressive generation

        Args:
            input_ids (Optional[torch.LongTensor], optional): Input tensor of shape `(batch_size, seq_length)`.
            attention_mask (Optional[torch.FloatTensor], optional): Attention mask tensor of shape `(batch_size, seq_length)`.

        Returns:
            torch.Tensor: A tensor `(batch_size, seq_length)` of predicted token ids
        """
        # list of tensors (num_features, batch_size, seq_length, feature_size)
        _, decoder_feature_output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        # tensor of size (batch_size, seq_length) that will start with seq_length=1
        current_generated_tokens = self.greedy_decode(decoder_feature_output)

        # Keep generating until every seq is done or we hit max length
        while (not (current_generated_tokens[:, -1] == self.decoder_pad_token_id).all()
               and current_generated_tokens.shape[1] < self.decoder_max_length):
            _, decoder_feature_output = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=current_generated_tokens)

            new_tokens = self.greedy_decode(decoder_feature_output)
            # Just add the last token predictions
            new_tokens = new_tokens[..., -1:]
            current_generated_tokens = torch.cat((current_generated_tokens, new_tokens), dim=1)

        return current_generated_tokens
