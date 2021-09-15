from flask import Flask, render_template, Response, request, session
from werkzeug.utils import redirect
from pathlib import Path
import time
from pathlib import Path
import os

from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

from annotation import detectAndDisplay
from model.FER2013_VGG19.VGG import VGG

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG('VGG19')
checkpoint = Path(__file__).parent / "model" / "FER2013_VGG19" / "PrivateTest_model.t7"
model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
trans = transforms.Compose([
    transforms.Resize((48, 48)),   
    transforms.ToTensor(),
])

uri = None
name = None

camera = cv2.VideoCapture(uri)
# uri: http://192.168.0.44:81/stream

root_dir = Path(__file__).parent

cut_size = 44

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = VGG('VGG19')
checkpoint = torch.load(root_dir / 'model' / 'FER2013_VGG19' / 'PrivateTest_model.t7')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

frames = []
temp_frame = None

# cv2.putText
# font = cv2.FONT_HERSHEY_SIMPLEX     
# org = [10, 50]
# fontScale = 0.7
# color = (255, 255, 255)
# thickness = 2

def gen_frames(camera):
    font = cv2.FONT_HERSHEY_SIMPLEX     
    org = [10, 30]
    fontScale = 0.7
    text_color = (255, 255, 255)
    face_border_color = (255, 255, 255)
    thickness = 2

    if not camera.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        time.sleep(0.05)
        success, frame = camera.read()
        if not success:
            print('--(!) No captured frame -- Break!')
            break

        temp_frame = frame

        frame, faceROI, faces = detectAndDisplay(frame)
        if faceROI is not None:
            faceROI = cv2.resize(faceROI, dsize=(48, 48), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            faceROI = faceROI[:, :, np.newaxis]
            faceROI = np.concatenate((faceROI, faceROI, faceROI), axis=2)
            faceROI = Image.fromarray(faceROI)
            inputs = transform_test(faceROI)

            ncrops, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
            outputs = net(inputs)
            outputs_avg = outputs.view(ncrops, -1).mean(0)

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
            score = score.tolist()
            score = list(map(lambda x: '{:.3f}'.format(x), score))

            score_dict = dict(zip(class_names, score))

            for key in score_dict.keys():
                if class_names[predicted] == key:
                    text_color = (50, 255, 50)
                cv2.putText(frame, key, org, font, fontScale, text_color, thickness, cv2.LINE_AA)
                org[0] = 150
                cv2.putText(frame, score_dict[key], org, font, fontScale, text_color, thickness, cv2.LINE_AA)
                org[1] += 30
                org[0] = 10
                text_color = (255, 255, 255)
            org = [10, 30]

            if class_names[predicted] == 'Happy':
                face_border_color = (50, 255, 50)

            (x,y,w,h) = faces[0]
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), face_border_color, 2)
            face_border_color = (255, 255, 255)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    if uri is None:
        return redirect('/register')
    return redirect('/stream')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global uri, name
        uri = request.values.get('uri')
        name = request.values.get('name')
        if uri == '0':
            uri = 0
        return redirect('/')
    return render_template('register.html')

@app.route('/stream')
def stream():
    if uri is None:
        return redirect('/register')
    return render_template('stream.html', uri=uri, name=name)

@app.route('/video_feed')
def video_feed():
    global uri
    camera = cv2.VideoCapture(uri)
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
