import f_Face_info
import cv2
import time
import os
import imutils
import argparse
from moviepy.editor import *
import librosa
from audio_gender_detection import f_test_gender

parser = argparse.ArgumentParser(description="Face Info")
parser.add_argument('--input', type=str, default= 'video',
                    help="webcam, image or video")
parser.add_argument('--path_im', type=str,
                    help="path of image")
args = vars(parser.parse_args())

type_input = args['input']
if type_input == 'image':
    # ----------------------------- image -----------------------------
    #take data
    frame = cv2.imread(args['path_im'])
    #get frame info
    out = f_Face_info.get_face_info(frame)
    #visualize data
    res_img = f_Face_info.bounding_box(out,frame)
    cv2.imshow('Face info',res_img)
    cv2.waitKey(0)

if type_input == 'webcam':
    # ----------------------------- webcam -----------------------------
    cv2.namedWindow("Face info")
    cam = cv2.VideoCapture(0)
    while True:
        star_time = time.time()
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)
        
        #get frame info
        out = f_Face_info.get_face_info(frame)
        #visualize data
        res_img = f_Face_info.bounding_box(out,frame)

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('Face info',res_img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
        
if type_input == 'video':
    # ----------------------------- video -----------------------------
    cv2.namedWindow("Face info")
    cam = cv2.VideoCapture(args['path_im'])
   
    path_librosa = '"' + args['path_im'] + '"'
    video = VideoFileClip(path_librosa)
    
    audio = video.audio
    # Replace the parameter with the location along with filename
    
    filename, file_extension = os.path.splitext(args['path_im'])
    audio_path = '"' + filename + '.wav"'
    
    audio.write_audiofile(audio_path)
    
    
    audio, sr = librosa.load(filename + '.wav')
    
    audio_detector = f_test_gender.get_gender_from_audio(sr,audio)
    
    while (cam.isOpened()):
        star_time = time.time()
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame, 
                    'GENDER FROM AUDIO: ' + audio_detector, 
                    (10, frame.Height-50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
        
        #get frame info
        out = f_Face_info.get_face_info(frame)
        #visualize data
        res_img = f_Face_info.bounding_box(out,frame)

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('Face info',res_img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    