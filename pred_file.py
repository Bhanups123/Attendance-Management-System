import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

model_emb = pickle.load(open('face_mo.pkl', 'rb'))
model_det = pickle.load(open('face_det.pkl', 'rb'))
model_cls = pickle.load(open('face_reco.pkl', 'rb'))
model_lab_enc = pickle.load(open('face_label.pkl', 'rb'))

def pred(image):

    pixels = np.asarray(image)
    results = model_det().detect_faces(pixels)
    
    x1, y1, width, height = results[0]['box']
    face = pixels[abs(y1):abs(y1)+height, abs(x1):abs(x1)+width]
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_arr = np.asarray(image)
    
    face_arr = face_arr.astype('float32')
    mean, std = face_arr.mean(), face_arr.std()
    face_arr = (face_arr - mean)/std
    face_arri = np.expand_dims(face_arr, axis = 0)
    
    emb = model_emb.predict(face_arri)
    
    emb_normalizer = Normalizer(norm = 'l2')
    emb_normed = emb_normalizer.transform(emb)
    
    z = model_cls.predict(emb_normed)
    name = model_lab_enc.inverse_transform(z)
    return name