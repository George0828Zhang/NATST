from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from omegaconf import II
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq import utils

@dataclass
class SpeechTransformerConfig(FairseqDataclass):
    # Convolutional subsampler
    # these 2 will be set by the data cfg
    input_feat_per_channel: Optional[int] = field(
        default=0, metadata={"help": "will be set by datacfg"}
    )
    input_channels: Optional[int] = field(
        default=0, metadata={"help": "will be set by datacfg"}
    )

    conv_kernel_sizes: Optional[str] = field(
        default="5,5", metadata={"help": "kernel sizes of Conv1d subsampling layers"}
    )
    conv_channels: Optional[int] = field(
        default=1024, metadata={"help": "# of channels in Conv1d subsampling layers"}
    )

    max_source_positions: Optional[int] = field(
        default=6000, metadata={"help": "max source sequence length"}
    )
    max_target_positions: Optional[int] = field(
        default=1024, metadata={"help": "max target sequence length"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "dropout probability"}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "dropout probability after activation in FFN."}
    )

    # encoder
    encoder_embed_dim: Optional[int] = field(
        default=512,
    )
    encoder_ffn_embed_dim: Optional[int] = field(
        default=2048,
    )
    encoder_layers: Optional[int] = field(
        default=12,
    )
    encoder_attention_heads: Optional[int] = field(
        default=8,
    )
    encoder_normalize_before: Optional[bool] = field(
        default=True, metadata={"help": 'apply layernorm before each encoder block'}
    )
    # encoder_learned_pos: Optional[bool] = field(
    #     default=False,
    # )

    # decoder
    decoder_embed_dim: Optional[int] = field(
        default=512,
    )
    decoder_ffn_embed_dim: Optional[int] = field(
        default=2048,
    )
    decoder_layers: Optional[int] = field(
        default=6,
    )
    decoder_attention_heads: Optional[int] = field(
        default=8,
    )
    decoder_normalize_before: Optional[bool] = field(
        default=True, metadata={"help": 'apply layernorm before each decoder block'}
    )
    decoder_learned_pos: Optional[bool] = field(
        default=False,
    )    
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )

    # embeddings
    # share_all_embeddings: Optional[bool] = field(
    #     default=False, metadata={"help": 'share encoder, decoder and output embeddings'
    #                              ' (requires shared dictionary and embed dim)'}
    # )
    share_decoder_input_output_embed: Optional[bool] = field(
        default=True, metadata={"help": 'share decoder input and output embeddings'}
    )
    layernorm_embedding: Optional[bool] = field(
        default=False, metadata={"help": 'add layernorm to embedding'}
    )
    no_scale_embedding: Optional[bool] = field(
        default=False, metadata={"help": 'if True, dont scale embeddings'}
    )
    no_token_positional_embeddings: Optional[bool] = field(
        default=False, metadata={"help": 'if True, dont scale embeddings'}
    )

    # load pretrain
    load_pretrained_encoder_from: Optional[str] = field(
        default=None, metadata={"help": 'model to take encoder weights from (for initialization)'}
    )

    # other/unused
    # checkpoint_activations: Optional[bool] = field(
    #     default=False, metadata={"help": 'checkpoint activations at each layer, which saves GPU '
    #                              'memory usage at the cost of some additional compute'}
    # )
    adaptive_softmax_cutoff: Optional[Any] = field(
        default=None, metadata={"help": ''}
    )
    adaptive_softmax_dropout: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    adaptive_input: Optional[bool] = field(
        default=False, metadata={"help": ''}
    )
    decoder_layerdrop: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    quant_noise_pq: Optional[float] = field(
        default=0, metadata={"help": ''}
    )

@dataclass
class MTTransformerConfig(SpeechTransformerConfig):
    # Convolutional subsampler
    # these 2 will be set by the data cfg
    input_feat_per_channel: Optional[int] = field(
        default=0, metadata={"help": "will be set by datacfg"}
    )
    input_channels: Optional[int] = field(
        default=0, metadata={"help": "will be set by datacfg"}
    )
        
    # other/unused
    encoder_learned_pos: Optional[bool] = field(
        default=False,
    )
    encoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": 'path to pre-trained encoder embedding'}
    )
    decoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": 'path to pre-trained decoder embedding'}
    )

    # embeddings
    share_all_embeddings: Optional[bool] = field(
        default=False, metadata={"help": 'share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)'}
    )

    # other/unused
    checkpoint_activations: Optional[bool] = field(
        default=False, metadata={"help": 'checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute'}
    )
    offload_activations: Optional[bool] = field(
        default=False, metadata={"help": 'checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.'}
    )
    adaptive_softmax_cutoff: Optional[Any] = field(
        default=None, metadata={"help": ''}
    )
    adaptive_softmax_dropout: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    adaptive_input: Optional[bool] = field(
        default=False, metadata={"help": ''}
    )
    tie_adaptive_weights: Optional[bool] = field(
        default=False, metadata={"help": ''}
    )

    encoder_layerdrop: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    decoder_layerdrop: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None, metadata={"help": ''}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None, metadata={"help": ''}
    )
    no_cross_attention: Optional[bool] = field(
        default=False, metadata={"help": ''}
    )
    cross_self_attention: Optional[bool] = field(
        default=False, metadata={"help": ''}
    )
    quant_noise_pq: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    quant_noise_pq_block_size: Optional[int] = field(
        default=8, metadata={"help": ''}
    )
    quant_noise_scalar: Optional[float] = field(
        default=0, metadata={"help": ''}
    )
    