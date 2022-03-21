import cv2
import time 

# image_path = "image/avt.jpg"
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# img = cv2.imread(image_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# while True:

#     faces = face_detector.detectMultiScale(img_gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     cv2.imshow('Frame', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cam = cv2.VideoCapture(1)
count = 0
while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    time.sleep(0.5)
    for (x, y, w, h) in faces:
        cut_faces = cv2.resize(frame[y+2: y+h-2, x+2: x+w-2], (100,100))
        cv2.imwrite('img_faces/face_{}.jpg'.format(count), cut_faces)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 2)

        count += 1
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()