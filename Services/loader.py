import os
import tensorflow as tf
import tensorflow_hub as hub
from Classes.vit_layers import SelfAttentionLayer, TransformerDecoderLayer, PositionalEncoding
from . import constants as cs
from tf_keras.models import load_model, Model
from tf_keras.layers import Input, Add, Concatenate
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

print("Loading Encoder ...")
# Create the encoder model
encoder = Model(inputs=caption_model.get_layer('input_1').input,
                outputs=caption_model.get_layer('tf.expand_dims').output)
print("Encoder Model Created Successfully")


print("Loading Decoder ...")
# Create the decoder model
caption_input = caption_model.get_layer('input_2').input
position_input = caption_model.get_layer('input_3').input
image_features_input = Input(shape=(1, 384), name='image_features_input')  # matches output of encoder

# Decoder subgraph
token_emb = caption_model.get_layer('embedding')(caption_input)
pos_enc = caption_model.get_layer('positional_encoding')(position_input)
x = token_emb + pos_enc

# Concatenate with image features
x = Concatenate(axis=1)([image_features_input, x])  # concat on sequence axis

# Pass through decoder layers
x = caption_model.get_layer('transformer_decoder_layer')(x)
x = caption_model.get_layer('transformer_decoder_layer_1')(x)
x = caption_model.get_layer('transformer_decoder_layer_2')(x)
x = caption_model.get_layer('transformer_decoder_layer_3')(x)

# Final output layer
output = caption_model.get_layer('dense_8')(x)

decoder = Model(inputs=[caption_input, position_input, image_features_input], outputs=output)
print("Decoder Model Created Successfully")