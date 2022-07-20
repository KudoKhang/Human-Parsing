import os.path

import cv2
import numpy as np
import argparse
from inference import HumanParsing

Human_parsing_predictor = HumanParsing()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="src/image_test/g2.jpg", help="Path to input")
    parser.add_argument("--tattoo", type=str, default="src/samples/3.png", help="Path to input")
    parser.add_argument("--save", action="store_false")
    parser.add_argument("--plot", action="store_false")
    parser.add_argument("--savedir", type=str, default=".")
    args = parser.parse_args()
    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("🎵 hhey, arguments are here if you need to check 🎵")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()
    return args

def image(path_img, path_tattoo, is_save=True, is_plot=True, savedir='.'):
    img = Human_parsing_predictor.run(path_img, path_tattoo)
    if is_save:
        path = os.path.join(savedir, 'result.png')
        cv2.imwrite(path, img)
        print('Save result 👉: ', path)
    if is_plot:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
    return img

def video(path_video):
    print('Processing video... \nPlease wait...')
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 30
    out = cv2.VideoWriter('results_' + path_video.split('/')[-1], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)


    while True:
        _, frame = cap.read()
        try:
            frame = Human_parsing_predictor.run(frame)
            frame = np.array(frame)
            out.write(frame)
        except:
            out.release()
            break
    out.release()
    print('Done!')


def webcam():
    print("Using webcam, press q to exit, press s to save")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        start = time.time()
        frame = Human_parsing_predictor.run(frame)
        frame = np.array(frame)
        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS : " + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow('Prediction', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite('image_out/' + str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break
