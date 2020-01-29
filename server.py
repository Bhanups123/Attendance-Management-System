from flask import Flask, request
from pred_file import pred, graph, sess
from tensorflow.python.keras.backend import set_session
from pymongo import MongoClient
import datetime
from bson.json_util import dumps

cluster = MongoClient("mongodb+srv://bhanu:12344321@cluster0-jrob2.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["Cluster0"]
collection = db["record"]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	global sess
	global graph
	with graph.as_default():
		set_session(sess)
		nparr = np.fromstring(request.data, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		name = pred(img)[0]
		date_time = datetime.datetime.now().strftime("%c")
		data = {"name": name, "date & time": date_time}
		collection.insert_one(data)
		return json.dumps({'msg': '√ Marked Successfully', 'name': name})
	
@app.route('/show_record')
def record():
	data_all = collection.find()
	return dumps(data_all)

app.run(debug = False, host='127.0.0.1', port=8000)