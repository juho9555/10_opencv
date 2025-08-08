import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. OCR 리더 생성 (한글 + 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# 2. 이미지 불러오기
# result = reader.readtext('chinese_tra.jpg')
img_path = '../img/children_sign.jpg'
img = cv2.imread(img_path)

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 3. 이미지 텍스트 인식
result = reader.readtext(img_path)
print(result)

# 5. 출력 오류 수정
FIX_TARGETS = {'어어린이보호': '어린이보호', '(어린이보호구예':'어린이보호구역'}

def fix_duplicates_with_dict(text):
    if text in FIX_TARGETS:
        return FIX_TARGETS[text]
    return text

# 4. 인식된 텍스트 확인해보기
THRESHOLD = 0.06 # 신뢰도 0.06
for bbox, text, conf in result:
    if conf >= THRESHOLD:
        fixed_text = fix_duplicates_with_dict(text)
        print(f'{fixed_text} (conf: {conf:.2f})')

        # numpy array로 변환 (정수형 좌표)
        pts = np.array(bbox, dtype=np.int32)

        # 기울어진 박스 그리기
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 한 번에 표시하기 위해 map과 tuple을 사용 (리스트이므로 튜플로 변환)
        #cv2.rectangle(img, pt1=tuple(map(int, bbox[0])), pt2=tuple(map(int, bbox[2])), color=(0, 255, 0), thickness=2)

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

