import models
import numpy as np
import cv2
import torch
import math
from torchvision import transforms

def model():
    model_path = 'model/CoolNetFinal'
    if model_path == 'model/CoolNetFinal':
        m = models.CoolNet()
    else:
        m = models.VGG()
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    return m

def mask_image(im):
    r = 128
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            xn = x-128
            yn = y-128
            if math.sqrt(xn*xn + yn*yn) >= r:
                im[x][y] = [255, 255, 255]

def transform(im):
    im = cv2.resize(im, (256, 256))
    #mask_image(im)
    #cv2.imwrite('face.jpg', im)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return t(im).unsqueeze(0)

def show_label(im, t, loc):
    t = t.view(-1).item()
    #print(t)
    p = "Male" if t >= 0 else "Female"
    args = [im, p, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2]
    cv2.putText(*args)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print(cap.get(3), cap.get(4))
    cap.set(3, 640)
    cap.set(4, 360)
    cascPath = 'lbpcascade_frontalface.xml'
    #cascPath = 'haar.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    m = model()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            subimage = frame[y:y+h, x:x+w]
            if subimage.size == 0:
                continue
            show_label(frame, m(transform(subimage)), (x, max(y - 5, 0)))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
