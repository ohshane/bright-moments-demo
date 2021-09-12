from flask import Flask, render_template, Response, request, session
from werkzeug.utils import redirect
from pathlib import Path

from PIL import Image
import cv2
import torch
from torchvision.transforms import transforms

from annotation import detectAndDisplay
from model.FER2013_VGG19 import VGG

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG()
checkpoint = Path(__file__).parent / "FER2013_VGG19" / "PrivateTest_model.t7"
model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
trans = transforms.Compose([
    transforms.Resize((48, 48)),   
    transforms.ToTensor(),
])

uri = None
name = None

camera = cv2.VideoCapture(uri)

def gen_frames(camera):
    if not camera.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        time.sleep(0.1)
        success, frame = camera.read()
        if not success:
            print('--(!) No captured frame -- Break!')
            break

        frame, faceROI = detectAndDisplay(frame)
        if faceROI is not None:
            faceROI = cv2.resize(faceROI, dsize=(48, 48), interpolation=cv2.INTER_LINEAR)

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
