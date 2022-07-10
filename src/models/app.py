import os
from flask import Flask,request,jsonify
import pickle
import subprocess
# import argparse
# parser = argparse.ArgumentParser()
import run_pose as rp
app = Flask(__name__)
# parser.add_argument('--image',type=str,help="add the path of the image file")
# parser.add_argument('--height',type=float,help="Actual Height of the Person in cm")
# args = parser.parse_args()
size = []
@app.route("/",methods = ['POST'])
def predict():
    b_i = str(request.values.get('image'))
    b_h = request.values.get('height')
    # print(b_i,b_h)
    # print(b_i)
    shoulder_final, waist_final, chest_final, neck_final, length_of_cloth, sleeve_length_final = rp.main(body_image=b_i,body_height=b_h)
    # shoulder_final, waist_final, chest_final, neck_final, length_of_cloth, sleeve_length_final
    data = {
        "shoulder":shoulder_final,
        "waist":waist_final,
        "chest":chest_final,
        "neck":neck_final,
        "Cloth_length":length_of_cloth,
        "sleeve_length":sleeve_length_final
    }
    return jsonify(data)

if __name__=='__main__':
    app.run(debug=True)