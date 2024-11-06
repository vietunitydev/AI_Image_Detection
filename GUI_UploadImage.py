import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import image_detector


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Hiển thị ảnh đã chọn lên GUI
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Lưu đường dẫn ảnh đã chọn
        app.selected_image = file_path

# Hàm để chạy dự đoán và hiển thị kết quả
def run_prediction():
    if hasattr(app, 'selected_image'):
        image_path = app.selected_image
        model_path = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Image_Classifier/trained_model.keras'

        prediction, confidence = image_detector.classify_image(image_path, model_path)

        # Hiển thị kết quả dự đoán trên màn hình
        result_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
        messagebox.showinfo("Result", f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
    else:
        messagebox.showerror("Error", "Please upload an image first.")

# Tạo cửa sổ GUI chính
app = tk.Tk()
app.title("Image Classifier")

# Nút để upload ảnh
upload_button = tk.Button(app, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

# Label để hiển thị ảnh đã upload
label_image = tk.Label(app)
label_image.pack(pady=10)

# Nút để chạy dự đoán
predict_button = tk.Button(app, text="Classify Image", command=run_prediction)
predict_button.pack(pady=20)

# Label để hiển thị kết quả dự đoán
result_label = tk.Label(app, text="Prediction: -\nConfidence: -", font=("Arial", 14))
result_label.pack(pady=10)

# Nút OK để tiếp tục upload ảnh mới
ok_button = tk.Button(app, text="OK", command=lambda: label_image.config(image=None))
ok_button.pack(pady=20)

# Chạy ứng dụng
app.mainloop()