import tensorflow as tf
import tf_keras

class PositionalEncoding(tf_keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # we’ll use this constant for the denominator exponent
        self._inv_freq = 1.0 / tf.pow(10000.0, (2 * tf.range(d_model, dtype=tf.float32) // 2) / tf.cast(d_model, tf.float32))

    def call(self, inputs):
        """
        inputs: int tensor of shape (batch_size, seq_len)
        """
        # shape: (seq_len,)
        seq_len = tf.shape(inputs)[1]
        pos = tf.cast(tf.range(seq_len), tf.float32)  # (seq_len,)

        # outer product: (seq_len, d_model)
        angles = tf.expand_dims(pos, -1) * tf.expand_dims(self._inv_freq, 0)

        # apply sin on even indices, cos on odd indices
        # exactly the same decision logic as your tf.where with mod 2
        pos_encoding = tf.where(
            tf.cast(tf.range(self.d_model) % 2, tf.bool)[tf.newaxis, :],
            tf.cos(angles),
            tf.sin(angles)
        )
        # expand to (1, seq_len, d_model) and tile to batch
        batch_size = tf.shape(inputs)[0]
        pos_encoding = tf.tile(pos_encoding[tf.newaxis, ...], [batch_size, 1, 1])
        return pos_encoding

class SelfAttentionLayer(tf_keras.layers.Layer):
    """ Defines the computations in the self attention layer """
    
    def __init__(self, d):        
        super(SelfAttentionLayer, self).__init__()
        # Feature dimensionality of the output
        self.d = d
    
    def build(self, input_shape):
        # Query weight matrix
        self.Wq = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )        
        # Key weight matrix
        self.Wk = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )
        # Value weight matrix
        self.Wv = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )
    
    def call(self, q_x, k_x, v_x, mask=None):
        
        q = tf.matmul(q_x,self.Wq) #[None, t, d]
        k = tf.matmul(k_x,self.Wk) #[None, t, d]
        v = tf.matmul(v_x,self.Wv) #[None, t, d]
        
        # Computing the final output
        h = tf_keras.layers.Attention(causal=True)([
            q, #q
            v, #v
            k, #k
        ], mask=[None, mask]) # [None, t, t] . [None, t, d] => [None, t, d]
        
        return h
    
    
class TransformerDecoderLayer(tf_keras.layers.Layer):
    """ The Decoder layer """
    
    def __init__(self, d, n_heads):
        super().__init__()
        
        # Feature dimensionality
        self.d = d
        
        # Dimensionality of a head
        self.d_head = int(d/n_heads) 
        
        # Number of heads
        self.n_heads = n_heads
        
        # Actual attention heads
        self.attn_heads = [SelfAttentionLayer(self.d_head) for _ in range(self.n_heads)]
        
        # Fully connected layers
        self.fc1_layer = tf_keras.layers.Dense(512, activation='relu')
        self.fc2_layer = tf_keras.layers.Dense(d)
        
        self.add_layer = tf_keras.layers.Add()
        self.norm1_layer = tf_keras.layers.LayerNormalization()
        self.norm2_layer = tf_keras.layers.LayerNormalization()
        
    
    def _compute_multihead_output(self, x):
        """ Computing the multi head attention output"""
        outputs = [head(x, x, x) for head in self.attn_heads]            
        outputs = tf.concat(outputs, axis=-1)
        return outputs
        
    def call(self, x):
        
        
        # Multi head attention layer output
        h1 = self._compute_multihead_output(x)
        
        h1_add = self.add_layer([x, h1])
        h1_norm = self.norm1_layer(h1_add)
        
        # Fully connected outputs
        h2_1 = self.fc1_layer(h1_norm)
        h2_2 = self.fc2_layer(h2_1)
        
        h2_add = self.add_layer([h1, h2_2])
        h2_norm = self.norm2_layer(h2_add)
        
        
        return h2_norm