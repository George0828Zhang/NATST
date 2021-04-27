import logging

from fairseq.models import (
    register_model, 
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
)

from .transformer_configs import MTTransformerConfig
logger = logging.getLogger(__name__)


@register_model("mt_transformer", dataclass=MTTransformerConfig)
class MTTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        *** change source dict to target dict ***
        """

        # make sure all arguments are present in older models
        # base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        src_dict, tgt_dict = task.target_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)