#Convolutional neural network (CNN) model for image classification trained on a NVIDIA RTX 2060. Code loads the training and #testing data, constructs the CNN model with specific layers and activations, performs training on the GPU, and finally saves #the trained model along with the accuracy results.
import pickle
import numpy as np

import tensorflow as tf
from keras import layers
from tqdm import tqdm
import pickle as pkl
import os
import gc


def check_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Kiểm tra cấu trúc dữ liệu của file pickle
    if 'data' in data and 'labels' in data:
        image_data = data['data']
        labels = data['labels']

        # Kiểm tra xem image_data có phải là mảng Numpy không
        if isinstance(image_data, np.ndarray):
            print(f"Image data is a numpy array with shape: {image_data.shape}")

            # Kiểm tra các đặc tính của mảng
            if image_data.ndim == 4:  # (batch_size, height, width, channels)
                print("Data format is correct for a batch of images.")
            elif image_data.ndim == 3 and image_data.shape[2] == 3:
                print("Data format is correct for a single RGB image.")
            else:
                print("Unexpected image data shape:", image_data.shape)
        else:
            print("Image data is not a numpy array.")

        # Kiểm tra xem labels có phải là mảng Numpy không
        if isinstance(labels, np.ndarray):
            print(f"Labels are a numpy array with shape: {labels.shape}")
        else:
            print("Labels are not a numpy array.")
    else:
        print("File does not contain expected 'data' and 'labels' keys.")

# Define the CNN model
def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(245, 255, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model

def train_model(model, train_data, test_data):
    print(f'---(Log) Start train .... ')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    results = []
    for epoch in range(10):
        total = 0
        print(f'---(Log) With epoch {epoch}')

        for batch in train_data:
            train_X, train_Y = batch['data'], batch['labels']
            test_X, test_Y = test_data['data'], test_data['labels']

            print(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}")
            if train_X is None or train_Y is None:
                print("Found None values in train data!")

            print(f"test_X: {test_X.shape}, test_Y: {test_Y.shape}")
            if test_X is None or test_Y is None:
                print("Found None values in test data!")

            # Train model
            history = model.fit(train_X, train_Y, epochs=1, validation_data=(test_X, test_Y))
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Save results
            results.append([history.history['val_accuracy'][0], history.history['accuracy'][0]])
    
    return results

# Load test set
with open('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/test_batches/test_batch.pickle', 'rb') as f:
    test_data = pkl.load(f)
# check_pickle_file('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/test_batches/test_batch.pickle')

# Load train set
train_data = []
batch_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/train_batches/'

valid_extensions = ['.pickle']

for batch in os.listdir(batch_path):
    if os.path.splitext(batch)[1].lower() in valid_extensions:
        # print(f'---(Log) List train patch {batch}')
        with open(batch_path + batch, 'rb') as f:
            # check_pickle_file(batch_path + batch)
            train_data.append(pkl.load(f))

# Create and train the model
model = create_model()
results = train_model(model, train_data, test_data)

# Save the trained model
model.save('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras')


# Save the results
with open('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/accuracy.pickle', 'wb') as f:
    pkl.dump(results, f)

