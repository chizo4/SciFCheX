'''
--------------------------------------------------------------
FILE:
    pipeline/training/tools/transformer.py

INFO:
    The script implements tools associated with training
    transformers for AR / RS / LP tasks. For instance, it
    allows model selection based on CLI args.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

from enum import Enum
import argparse

class Transformer(Enum):
    '''
    -------------------------
    Transformer enum class to map transformer to their HF checkpoint.
    -------------------------
    '''
    BERT          = 'bert-base-uncased'
    SCIBERT       = 'allenai/scibert_scivocab_uncased'
    BIOBERT_BASE  = 'dmis-lab/biobert-base-cased-v1.1'
    BIOBERT_LARGE = 'dmis-lab/biobert-large-cased-v1.1'
    ROBERTA_LARGE = 'roberta-large'

def map_transformer() -> tuple:
    '''
    Map a transformer's name into its checkpoint, for training.

        Returns:
        -------------------------
        (transformer_checkpoint, transformer_cli) : tuple
            Tuple of transformer name mapped to specific checkpoint from HuggingFace,
            and its raw CLI selection (for path save).
    '''
    # Transformer name (from CLI) to be mapped to specific checkpoint.
    args = set_args()
    transformer_cli = args.model
    transformer_map = {
        'bert': Transformer.BERT,
        'scibert': Transformer.SCIBERT,
        'biobert_base': Transformer.BIOBERT_BASE,
        'biobert_large': Transformer.BIOBERT_LARGE,
        'roberta_large': Transformer.ROBERTA_LARGE,
    }
    transformer_checkpoint = transformer_map[transformer_cli].value
    if not transformer_checkpoint:
        raise ValueError('Unsupported Transformer! Please make sure the model name is supported.')
    return (transformer_checkpoint, transformer_cli)

def set_args() -> argparse.Namespace:
    '''
    Handle parsing CLI args associated with training.

        Returns:
        -------------------------
        args : argparse.Namespace
            Object contains pre-processed CLI args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True
    )
    return parser.parse_args()
