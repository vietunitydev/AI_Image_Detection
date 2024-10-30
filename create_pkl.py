from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request
import ssl

# Bỏ qua SSL verification
context = ssl._create_unverified_context()

# Đường dẫn driver
PATH = '/Users/sakai/Downloads/chromedriver-mac-arm64-2/chromedriver'
driver = webdriver.Chrome(service=Service(PATH))

# Thư mục lưu ảnh
HUMAN = '/Users/sakai/VIET_Working/STUDY_WORK/Ky5/Python/Human'
urls = ['https://www.google.com/search?q=art+graphic+design&tbm=isch']

# Danh mục ảnh
categories = ['a_graphic_design_art']

# Tải và lưu ảnh từ mỗi URL
for idx, url in enumerate(urls):
    driver.get(url)
    time.sleep(2)  # Đợi trang tải xong

    # Cuộn trang để tải ảnh
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(10):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)

    # Tìm tất cả các ảnh sau khi cuộn
    images = driver.find_elements(By.TAG_NAME, 'img')
    os.makedirs(os.path.join(HUMAN, categories[idx]), exist_ok=True)

    # Duyệt và tải từng ảnh
    for i, img in enumerate(images):
        src = img.get_attribute('src')

        # Kiểm tra thêm thuộc tính 'data-src' nếu 'src' rỗng
        if not src:
            src = img.get_attribute('data-src')

        if src and 'http' in src:
            try:
                file_path = os.path.join(HUMAN, categories[idx], f'image_{i}.jpg')

                # Tải ảnh với urlopen và bỏ qua xác minh SSL
                with urllib.request.urlopen(src, context=context) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())

                print(f"Đã tải xong ảnh {i} từ '{categories[idx]}' - Lưu tại: {file_path}")

            except Exception as e:
                print(f"Lỗi khi tải ảnh {i}: {e}")

driver.quit()
