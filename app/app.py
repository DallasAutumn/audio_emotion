import json
import os
import random

import torch
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from flask_bootstrap import Bootstrap
from flask_cors import CORS
from werkzeug.utils import secure_filename

from models import CNN

from ..extract_features import *
from ..transforms import *

# 全局常量定义
UPLOAD_FOLDER = "/run/media/dallasautumn/data/duan-qiu-yang/audio_emotion/app/backend/uploads"
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
DEBUG = True

app = Flask(__name__,
            static_folder="./backend/static",
            template_folder="./backend/templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bootstrap = Bootstrap(app)

# 处理跨域请求
cors = CORS(app, resources={r"/*": {"origins": "*"}})


class2index = json.load(open("class_index.json"))
index2class = {v: k for k, v in class2index.items()}

# 在此处声明，保证模型只在app初始化时加载
device = torch.device('cpu')
model = CNN()
model.load_state_dict(torch.load('pickles/cnn.pkl'))
model.to(device=device)

trans = Compose([ToTensor(),
                 PaddingSame2d(seq_len=224, value=0)])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_audio(audio_bytes):
    feat = get_mfcc(audio_bytes)
    feat = trans(feat)
    return feat


def get_prediction(audio_bytes):
    tensor = transform_audio(audio_bytes).unsqueeze(dim=0)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx, index2class[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         flash('No file part')
    #         return redirect(request.url)
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        class_id, class_name = get_prediction(filepath)
        resp = jsonify(
            {
                'class_id': class_id,
                'class_name': class_name
            }
        )

        print(resp.json)
        return resp


# 所有路由全部重定向到index（单页应用）
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=DEBUG)
