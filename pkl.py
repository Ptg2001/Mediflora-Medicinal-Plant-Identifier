import pickle
from keras.models import load_model

# Step 1: Load the Keras model from H5 file
model = load_model('model/model_sev.h5')  # Replace 'your_model.h5' with the path to your H5 file

# Step 2: Save the model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model successfully converted from H5 to PKL format.")
