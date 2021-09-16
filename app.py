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
# from torchsummary import summary

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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

target_layer = model.features.__getitem__(0)

gradcam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
scorecam = ScoreCAM(model=model, target_layer=target_layer, use_cuda=True)
gradcamplusplus = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)
ablationcam = AblationCAM(model=model, target_layer=target_layer, use_cuda=True)
xgradcam = XGradCAM(model=model, target_layer=target_layer, use_cuda=True)
eigencam = EigenCAM(model=model, target_layer=target_layer, use_cuda=True)
cams = [gradcam, scorecam, gradcamplusplus, ablationcam, xgradcam, eigencam]

uri = None
name = None
noise = False

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
# print(summary(net, (3, 44, 44)))

frames = []
temp_frame = None

def gen_frames(camera):
    # for cv2.putText
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
        # time.sleep(1)
        success, frame = camera.read()
        if not success:
            print('--(!) No captured frame -- Break!')
            break

        frame, faceROI, faces = detectAndDisplay(frame, noise=noise)
        if faceROI is not None:
            gradcam_input = None
            (x,y,w,h) = faces[0]
            
            faceROI = cv2.resize(faceROI, dsize=(48, 48), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            faceROI = faceROI[:, :, np.newaxis]
            faceROI = np.concatenate((faceROI, faceROI, faceROI), axis=2)
            faceROI = Image.fromarray(faceROI)
            inputs = transform_test(faceROI)

            ncrops, channel, height, width = np.shape(inputs)

            inputs = inputs.view(-1, channel, height, width)
            inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
            outputs = net(inputs)
            outputs_avg = outputs.view(ncrops, -1).mean(0)

            # grayscale_cam = gradcam(input_tensor=inputs)
            # grayscale_cam = grayscale_cam[0, :]

            # visualization_cam = show_cam_on_image(inputs, grayscale_cam)
            # print(visualization_cam)

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
            score = score.tolist()
            score = list(map(lambda x: '{:.3f}'.format(x), score))

            score_dict = dict(zip(class_names, score))

            # 3-class
            temp_dict = {}
            temp_dict['Happy'] = score_dict['Happy']
            temp_dict['Neutral'] = score_dict['Neutral']
            temp_dict['Others'] = '{:.3f}'.format(1 - (float(score_dict['Happy']) + float(score_dict['Neutral'])))
            score_dict = temp_dict
            

            for key in score_dict.keys():
                if class_names[predicted] == key:
                    text_color = (50, 255, 50)
                elif key == 'Others' and class_names[predicted] in ['Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']:
                    text_color = (50, 255, 50)
                cv2.putText(frame, key, org, font, fontScale, text_color, thickness, cv2.LINE_AA)
                org[0] = 110
                cv2.putText(frame, score_dict[key], org, font, fontScale, text_color, thickness, cv2.LINE_AA)
                cv2.rectangle(frame, (org[0]+90, org[1]-15), (org[0]+190, org[1]), (255,255,255), 2)
                cv2.rectangle(frame, (org[0]+90, org[1]-15), (org[0]+90+int(float(score_dict[key])*100), org[1]), (255,255,255), -1)
                score_dict[key] * 100
                org[1] += 30
                org[0] = 10
                text_color = (255, 255, 255)
            org = [10, 30]

            if class_names[predicted] == 'Happy':
                face_border_color = (50, 255, 50)

            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), face_border_color, 1)
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
        global uri, name, noise
        uri = request.values.get('uri')
        name = request.values.get('name')
        noise = request.form.get('noise')
        if uri == '0':
            uri = 0
        if noise == 'off':
            noise = False
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
