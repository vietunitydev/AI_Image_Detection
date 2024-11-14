import tensorflow as tf
from keras import layers
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os
import gc


def check_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

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
# Mô hình nhận đầu vào là các hình ảnh có kích thước 256 x 256 pixel với 3 kênh màu (RGB).
# layers.Conv2D(32, (3, 3), activation='relu'), : Tạo một lớp tích chập với 32 bộ lọc, mỗi bộ lọc có kích thước 3x3, và sử dụng hàm kích hoạt ReLU (Rectified Linear Unit) để tạo ra các đặc trưng từ đầu vào.
# layers.MaxPooling2D((2, 2)): Tạo một lớp lấy mẫu tối đa với kích thước 2x2 để giảm kích thước của đầu ra từ lớp tích chập, từ đó giúp giảm số lượng tham số và tính toán.
# thực hiện hai lần một cặp lớp Conv2D và MaxPooling, giúp tăng cường khả năng trích xuất đặc trưng của mô hình.
# layers.Flatten(): Làm phẳng đầu ra từ các lớp tích chập và lấy mẫu, chuyển đổi thành một mảng một chiều để chuẩn bị cho các lớp Dense.
# layers.Dense(64, activation='relu'): Một lớp dày với 64 nơ-ron và hàm kích hoạt ReLU.
# layers.Dense(3, activation='softmax'): Lớp đầu ra với 3 nơ-ron (tương ứng với 3 lớp mà bạn muốn phân loại) và hàm kích hoạt softmax, giúp xác định xác suất cho mỗi lớp

def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model

def train_model(model, train_data, test_data):
    print(f'---(Log) Start train .... ')

    # tạo một optimizer Adam với tốc độ học (learning rate) được chỉ định là 0.00005
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    # biên dịch sử dụng categorical_crossentropy làm hàm mất mát.
    # metrics=['accuracy'] được sử dụng để theo dõi độ chính xác của mô hình trong quá trình huấn luyện.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    results = []
    for epoch in tqdm(range(10)):
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

            # model.fit: Đây là phương thức dùng để huấn luyện mô hình Keras với dữ liệu huấn luyện
            # và gán giá trị cho các trọng số của mô hình dựa trên mất mát (loss) và độ chính xác (accuracy).

            # epochs=1:
            # Số lượng epoch là số lần mô hình sẽ duyệt qua toàn bộ dữ liệu huấn luyện.
            # Trong trường hợp này, bạn chỉ đặt epochs=1, có nghĩa là mô hình sẽ chỉ huấn luyện một lần qua toàn bộ dữ liệu huấn luyện.
            # Thường thì bạn sẽ tăng số lượng epoch này lên để cải thiện độ chính xác,
            # nhưng có thể sẽ cần theo dõi để tránh tình trạng quá khớp (overfitting).

            # validation_data=(test_X, test_Y):
            # validation_data: Là tham số cho phép bạn cung cấp dữ liệu kiểm tra (validation set) để đánh giá hiệu suất của mô hình sau mỗi epoch.
            # test_X: Dữ liệu đầu vào dùng để kiểm tra mô hình (không tham gia vào quá trình huấn luyện).
            # test_Y: Nhãn tương ứng với các mẫu trong test_X.
            # Việc cung cấp dữ liệu kiểm tra giúp bạn theo dõi độ chính xác và mất mát của mô hình trên dữ liệu chưa thấy trong quá trình huấn luyện.

            # Biến history sẽ chứa thông tin về quá trình huấn luyện, bao gồm giá trị mất mát (loss) và độ chính xác (accuracy) cho cả dữ liệu huấn luyện và kiểm tra qua mỗi epoch.
            # Bạn có thể sử dụng history.history để truy cập các thông tin này sau khi huấn luyện xong, ví dụ để vẽ biểu đồ hoặc phân tích hiệu suất.
            history = model.fit(train_X, train_Y, epochs=1, validation_data=(test_X, test_Y))

            # gc.collect() được gọi để thu gom bộ nhớ không còn được sử dụng.
            # tf.keras.backend.clear_session() giúp giải phóng tài nguyên của phiên Keras trước đó, giúp tránh lỗi khi tái sử dụng mô hình.
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Save results
            results.append([history.history['val_accuracy'][0], history.history['accuracy'][0]])

    return results

# Load test set
with open('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/test_batches/test_batch.pickle', 'rb') as f:
    test_data = pkl.load(f)

# Load train set
train_data = []
batch_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/train_batches/'

valid_extensions = ['.pickle']

for batch in os.listdir(batch_path):
    if os.path.splitext(batch)[1].lower() in valid_extensions:
        with open(batch_path + batch, 'rb') as f:
            train_data.append(pkl.load(f))


# Create and train the model
model = create_model()
results = train_model(model, train_data, test_data)

# Save the trained model
model.save('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras')


# Save the results
with open('/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/accuracy.pickle', 'wb') as f:
    pkl.dump(results, f)

