#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask

from torch import Tensor

from fairseq.models import (
    FairseqEncoderModel, 
    register_model, 
    register_model_architecture
)
from fairseq.models.transformer import (
    Embedding, 
    Linear,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel, 
    S2TTransformerEncoder as S2TTransformerEncoderProto
)

from .transformer_configs import SpeechTransformerConfig
from .nat_generate import generate

logger = logging.getLogger(__name__)

class S2TTransformerEncoder(S2TTransformerEncoderProto):
    """Enables encoder hidden states to be used in downstream modules.
    """
    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False,):
        """ Same as prototype but returns hidden states """
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model("full_encoder", dataclass=SpeechTransformerConfig)
class S2TTransformerEncoderModel(FairseqEncoderModel): #FairseqEncoderDecoderModel):
    def __init__(self, encoder, output_projection):
        super().__init__(encoder)
        self.output_projection = output_projection
        self.one_pass_decoding = True # must implement generate()

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        output_projection = nn.Linear(
            args.encoder_embed_dim, len(task.target_dictionary), bias=False
        )
        nn.init.normal_(
            output_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )

        encoder = cls.build_encoder(args)
        # decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, output_projection)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        logits = net_output[0]

        if torch.is_tensor(logits):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits_f = logits.float()
            if log_probs:
                lprobs = F.log_softmax(logits_f, dim=-1)
            else:
                lprobs = F.softmax(logits_f, dim=-1)
        else:
            raise NotImplementedError

        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        x = self.output_projection(encoder_out["encoder_out"][0])
        x = x.transpose(1,0) # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
        if len(encoder_out["encoder_padding_mask"])>0 else None
        return x, {"padding_mask": padding_mask, "encoder_out": encoder_out}

    def generate(self, src_tokens, src_lengths, blank_idx=0, **unused):
        return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)
