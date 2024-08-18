import tensorflow as tf
import pickle

# Load the .h5 model
model = tf.keras.models.load_model('model/model_sev.h5')

# Convert the model to a pickle file
with open('model_sev.pkl', 'wb') as f:
    pickle.dump(model, f)
