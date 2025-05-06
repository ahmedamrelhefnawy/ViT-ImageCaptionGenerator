import os
import tensorflow as tf
import tensorflow_hub as hub
from Classes.vit_layers import SelfAttentionLayer, TransformerDecoderLayer, PositionalEncoding
from . import constants as cs
from tf_keras.models import load_model
from tokenizers import Tokenizer


# Load the Tokenizer    
print("Loading Tokenizer")
tokenizer = Tokenizer.from_file(os.path.join(cs.models_dir, 'ViTBertTokenizer.json'))
print("Tokenizer Loaded Successfully")

# Load the model
print("Loading Model")
caption_model = load_model(os.path.join(cs.models_dir, 'ViTCaptionGenerator.keras'),
                           custom_objects={'KerasLayer': hub.KerasLayer,
                                           'SelfAttentionLayer': SelfAttentionLayer,
                                           'TransformerDecoderLayer': TransformerDecoderLayer,
                                           'PositionalEncoding': PositionalEncoding,
                                            },
                           safe_mode=False
                           )
print("Model Loaded Successfully")
