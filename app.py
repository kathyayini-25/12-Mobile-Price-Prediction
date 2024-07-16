from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = pickle.load(open('knn.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    battery_power = request.form['battery_power']
    blue = request.form['blue']
    clock_speed  = request.form['clock_speed']
    dual_sim = request.form['dual_sim']
    fc = request.form['fc']
    four_g = request.form['four_g']
    int_memory = request.form['int_memory']
    m_dep = request.form['m_dep']
    mobile_wt = request.form['mobile_wt']
    n_cores = request.form['n_cores']
    pc = request.form['pc']
    px_height = request.form['px_height']
    px_width = request.form['px_width']
    ram = request.form['ram']
    sc_h = request.form['sc_h']
    sc_w = request.form['sc_w']
    talk_time = request.form['talk_time']
    three_g = request.form['three_g']
    touch_screen = request.form['touch_screen']
    wifi = request.form['wifi']
    arr = np.array([[battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi]])
    arr=arr.astype(float)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)














