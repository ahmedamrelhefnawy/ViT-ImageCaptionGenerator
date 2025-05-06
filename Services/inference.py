import tensorflow as tf
import numpy as np

def generate_caption(model, image_input, tokenizer):
    # 2 -> [START]
    batch_tokens = np.repeat(np.array([[2]]), 1, axis=0)
    
    for i in range(74):
        if np.all(batch_tokens[:, -1] == 3):  # If [END] token is reached, stop generating
            break
            
        position_input = tf.repeat(tf.reshape(tf.range(i+1), [1, -1]), 1, axis=0)
        probs = model((image_input, batch_tokens, position_input)).numpy()
        batch_tokens = np.argmax(probs, axis=-1)
        
    predicted_text = []
    predicted_token_ids = batch_tokens.ravel()
    
    predicted_tokens = []
    for wid in predicted_token_ids:
        if wid == 2: # [START] token
            continue
        if wid == 3:  # [END] token
            break
        predicted_tokens.append(tokenizer.id_to_token(wid))
    
    predicted_text = " ".join([tok for tok in predicted_tokens])
    predicted_text = predicted_text.replace(" ##", "")

    return predicted_text