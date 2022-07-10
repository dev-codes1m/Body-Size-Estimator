    # To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
#import imutils
import time
from math import dist
body_image = "D:/sample6.jpg"
body_height = 176
def main(body_image,body_height):
    parser = argparse.ArgumentParser(
            description='This script is used to demonstrate OpenPose human pose estimation network '
                        'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                        'The sample and model are simplified and could be used for a single person on the frame.')
    parser.add_argument('--input', help='Path to input image.')
    parser.add_argument('--proto', help='Path to .prototxt')
    parser.add_argument('--model', help='Path to .caffemodel')
    parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                          'It could be (COCO, MPI) depends on dataset.')
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--person_height',type=float,help = 'Enter the Height of the Person')

    args = parser.parse_args()
    args.input = body_image
    args.person_height = float(body_height)
    args.proto = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    args.model = "pose/mpi/pose_iter_160000.caffemodel"
    args.dataset = "MPI"


    if args.dataset == 'COCO':
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                       "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                       ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                       ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                       ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                       ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    elif args.dataset=='MPI':
        #assert(args.dataset == 'MPI')
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                       "Background": 15 }

        # POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
        #               ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
        #               ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
        #               ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
        POSE_PAIRS = [["RShoulder","LShoulder"],["RHip","RWrist"],["LHip","LWrist"],["Head","RAnkle"],["Chest","RShoulder"],["Chest","LShoulder"],["Neck","RShoulder"],["Neck","LShoulder"],["RShoulder","RElbow"]]
    else:

        BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

        POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],
    ["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

    inWidth = args.width
    inHeight = args.height

    net = cv.dnn.readNetFromCaffe(args.proto, args.model)


    frame = cv.imread(args.input)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()

    # print("time is ",time.time()-start_t)
    # print(inp.shape)
    kwinName="Pose Estimation Demo: Cv-Tricks.com"
    cv.namedWindow(kwinName, cv.WINDOW_AUTOSIZE)
    #assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
        # print(points)
        body_dim = []

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame, str(idFrom), points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame, str(idTo), points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            # print(cv.norm(points[idFrom],points[idTo]))
            body_dim.append([points[idFrom],points[idTo]])
    # print(body_dim)
    # POSE_PAIRS = [["RShoulder", "LShoulder"], ["RHip", "RWrist"], ["LHip", "LWrist"], ["Head", "RAnkle"],
    #               ["Chest", "RShoulder"], ["Chest", "LShoulder"]]

    shoudler = ((abs(body_dim[0][0][0]-body_dim[0][1][0])**2 + abs(body_dim[0][0][1]-body_dim[0][1][1])**2))**0.5
    waistrx = (body_dim[1][0][0] + body_dim[1][1][0])/2
    waistry = (body_dim[1][0][1] + body_dim[1][1][1])/2
    waistlx = (body_dim[2][0][0] + body_dim[2][1][0])/2
    waistly = (body_dim[2][0][1] + body_dim[2][1][1])/2
    waist = ((abs(waistlx-waistrx)**2 + abs(waistly - waistry)**2)**0.5)
    height = ((abs(body_dim[3][0][0] - body_dim[3][1][0])**2 + abs(body_dim[3][1][0] - body_dim[3][1][1])**2)**0.5)
    chestrx = ((body_dim[4][0][0] + body_dim[4][1][0])/2)
    chestlx = ((body_dim[5][0][0] + body_dim[5][1][0])/2)
    chestly = ((body_dim[4][0][1] + body_dim[4][1][1])/2)
    chestry = ((body_dim[5][0][1] + body_dim[5][1][1])/2)
    neckrx = ((body_dim[6][0][0] + body_dim[6][1][0])/2)
    necklx = ((body_dim[7][0][0] + body_dim[7][1][0])/2)
    neckly = ((body_dim[6][0][1] + body_dim[6][1][1])/2)
    neckry = ((body_dim[7][0][1] + body_dim[7][1][1])/2)
    length = (abs(body_dim[1][0][0] - body_dim[6][0][0])**2 + abs(body_dim[1][0][1] - body_dim[6][0][1])**2)**0.5
    neck = ((abs(neckrx - necklx)**2 + abs(neckry - neckly)**2)**0.5)
    chest = ((abs(chestlx - chestrx)**2 + abs(chestly - chestry)**2)**0.5)
    sleeve_length = (abs(body_dim[8][0][0] - body_dim[8][1][0])**2 + abs(body_dim[8][0][1] - body_dim[8][1][1])**2)**0.5
    x = args.person_height/height # Multiplication Factor
    shoulder_final = shoudler*x
    waist_final = (waist*x)*3.14/1.49
    chest_final = (chest*x)*3.14/1.49
    neck_final = (neck/2)*x
    length_of_cloth = length*x
    sleeve_length_final = (sleeve_length/2)*x
    return shoulder_final,waist_final,chest_final,neck_final,length_of_cloth,sleeve_length_final
    # print(shoulder_final,waist_final,chest_final,neck_final,length_of_cloth,sleeve_length_final)

if __name__ == '__main__':
    main()
# if main() =="__main__":
#      main()


# print(shoulder_final)
# print(waist_final)
# print(chest_final)
# print(neck_final)
# print(length_of_cloth)
# print(sleeve_length_final)

# t, _ = net.getPerfProfile()
# freq = cv.getTickFrequency() / 1000
# cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
#
# cv.imshow(kwinName, frame)
# cv.imwrite('result_'+args.input,frame)