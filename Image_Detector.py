import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size):
    """Loads and preprocesses the image from the given path."""
    img = Image.open(image_path).resize(target_size)
    img_arr = np.array(img.convert("RGB")) / 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def classify_image(image_path, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)


    # Preprocess the input image
    img_array = preprocess_image(image_path, (256,256))

    # Perform prediction
    predictions = model.predict(img_array)

    print(predictions)

    # Assuming index 0 is human and index 1 is AI-generated
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(predicted_class)

    confidence = predictions[0][predicted_class]
    print(confidence)

    # Class labels
    class_labels = {0: "AI", 1: "HUMAN"}

    # Print results
    print(f"Prediction: {class_labels[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")

    return class_labels[predicted_class], confidence


# Test with an image path
# image_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Dataset/human/00200.png'
# model_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras'
# prediction, confidence = classify_image(image_path, model_path)

# /Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Dataset/ai_generated/000072.jpg
# /Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Dataset/human/00007.png