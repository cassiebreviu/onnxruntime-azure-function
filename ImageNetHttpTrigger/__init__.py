import logging
import azure.functions as func
import base64
import numpy as np
import cv2
import io
import onnxruntime as ort


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    img_base64 = req.params.get('img')
    if not img_base64:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            img_base64 = req_body.get('img')

    if img_base64:
        img = decode_base64(img_base64)
        model_path = './ImageNetHttpTrigger/resnet50v2.onnx'
        outputs = run_model(model_path, img)
        return func.HttpResponse(f"The image is a {map_outputs(outputs)}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a img_base64 in the query string or in the request body for a personalized response.",
             status_code=200
        )

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# decode base64 image
def decode_base64(data):
    img = base64.b64decode(data)
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2,0,1))
    img = img.reshape(1, 3, 224, 224)
    img = preprocess(img)
    return img

# run model on image
def run_model(model_path, img):
    #img = load_image(img_path)
    ort_sess = ort.InferenceSession(model_path)
    outputs = ort_sess.run(None, {'data': img})
    return outputs

#load text file as list
def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

# map mobilenet outputs to classes
def map_outputs(outputs):
    labels = load_labels('./ImageNetHttpTrigger/imagenet_classes.txt')
    return labels[np.argmax(outputs)]