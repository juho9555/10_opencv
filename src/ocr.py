import easyocr
import cv2
import matplotlib.pyplot as plt

reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# result = reader.readtext('chinese_tra.jpg')
img_path = '../img/chinese_tra.jpg'
img = cv2.imread(img_path)

# 1. 이미지 불러오기
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 2. 이미지 텍스트 인식
result = reader.readtext(img_path)
print(result)