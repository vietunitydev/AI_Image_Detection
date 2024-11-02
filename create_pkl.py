import numpy as np
import tensorflow as tf
from PIL import Image
import webp
import image

# Tải mô hình đã được huấn luyện
model = tf.keras.models.load_model('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras')

def preprocess_image(image_path, target_size):
    """Loads and preprocesses the image from the given path."""
    img = Image.open(image_path).resize(target_size)
    img = np.array(img.convert("RGB")) / 255.
    img = np.expand_dims(img, axis=0)
    return img

def predict_image_class(img_path):
    """Dự đoán lớp của ảnh: Con người hay AI."""
    # Chuẩn bị ảnh
    prepared_image = preprocess_image(img_path, (255,245))

    # Dự đoán lớp
    predictions = model.predict(prepared_image)

    # Chuyển đổi dự đoán thành nhãn
    predicted_label = np.argmax(predictions, axis=1)

    # Xác định nhãn tương ứng
    class_names = {0: "Con người", 1: "AI"}  # Điều chỉnh theo nhãn của bạn
    predicted_class = class_names.get(predicted_label[0], "Không xác định")

    return predicted_class

img_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Dataset/ai_generated/image_4.jpg'
result = predict_image_class(img_path)
print(f"Dự đoán: {result}")
