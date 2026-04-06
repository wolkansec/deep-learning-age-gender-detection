import cv2

# ---------------------------------------
# Model dosyaları ve parametreler
# ---------------------------------------
face_proto_file   = "model/opencv_face_detector.pbtxt"
face_model_file   = "model/opencv_face_detector_uint8.pb"
age_proto_file    = "model/age_deploy.prototxt"
age_model_file    = "model/age_net.caffemodel"
gender_proto_file = "model/gender_deploy.prototxt"
gender_model_file = "model/gender_net.caffemodel"


# Yaş tahmini için kullanılan ortalama değerler
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

# Yaş ve cinsiyet kategorileri
AGE_CATEGORIES = ['(0-2)','(4-6)','(8-12)','(15-18)','(25-32)','(38-43)','(48-53)','(60-100)']
GENDERS = ['Male','Female']

# Ağların Yüklenmesi
face_net   = cv2.dnn.readNet(face_model_file, face_proto_file)
age_net    = cv2.dnn.readNet(age_model_file, age_proto_file)
gender_net = cv2.dnn.readNet(gender_model_file, gender_proto_file)


# ---------------------------------------
# Yüzleri algılayan ve çerçeveleyen fonksiyon
# ---------------------------------------
def draw_faces(network, image, threshold=0.7):
    # Orijinal görüntüyü değiştirmemek için kopya oluştur
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    # OpenCV DNN blob formatına çevirme
    blob = cv2.dnn.blobFromImage(
        img_copy,
        scalefactor=1.0,
        size=(300,300),
        mean=[104,117,123],  # Ortalama değerler ile normalize
        swapRB=True,          # BGR → RGB dönüşümü
        crop=False
    )
    network.setInput(blob)
    detections = network.forward()
    
    boxes = []

    # Algılanan her yüz için koordinatları hesapla
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]  # Tahmin güven skoru
        if conf > threshold:
            x_start = int(detections[0,0,i,3] * w)
            y_start = int(detections[0,0,i,4] * h)
            x_end   = int(detections[0,0,i,5] * w)
            y_end   = int(detections[0,0,i,6] * h)

            boxes.append([x_start, y_start, x_end, y_end])

            # Yüzü kırmızı dikdörtgenle çiz
            thickness = max(1, round(h / 150))
            cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end),
                          (0,0,255), thickness, lineType=8)

    return img_copy, boxes

# ---------------------------------------
# Görüntüyü aç ve işleme
# ---------------------------------------
video_src = "test_files/test_image1.jpg"
cap = cv2.VideoCapture(video_src)
padding = 20  # Yüz çevresine ekstra boşluk eklemek için


while True:
    # Frame oku
    ret, frame = cap.read()
    if not ret:
        cv2.waitKey()
        break

    # Yüzleri algıla ve çerçevele
    output_img, detected_faces = draw_faces(face_net, frame)

    if not detected_faces:
        print("No faces were found!")

    for face_coords in detected_faces:
        x1, y1, x2, y2 = face_coords

        # Yüz ROI (Region of Interest) çıkar
        face_roi = frame[
            max(0, y1-padding):min(y2+padding, frame.shape[0]-1),
            max(0, x1-padding):min(x2+padding, frame.shape[1]-1)
        ]

        # Yaş ve cinsiyet tahmini için blob oluştur
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227,227), MODEL_MEAN, swapRB=False)

        # -----------------------
        # Cinsiyet tahmini
        # -----------------------
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDERS[gender_preds[0].argmax()]

        # -----------------------
        # Yaş tahmini
        # -----------------------
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_CATEGORIES[age_preds[0].argmax()]

        # Konsola yazdır
        print(f"Output: Gender: {gender}, Age: {age[1:-1]}")
        print("---------------------------------------")

        # Görüntüye yazdır
        cv2.putText(output_img, f"{gender}, {age}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

    # Sonuç görüntüyü göster
    cv2.imshow("Age & Gender Detection", output_img)

    # 'q' tuşu ile çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()