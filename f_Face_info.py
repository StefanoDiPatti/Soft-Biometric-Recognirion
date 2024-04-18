import cv2
import numpy as np
import face_recognition
from age_detection import f_my_age
from gender_detection import f_my_gender
from race_detection import f_my_race
from emotion_detection import f_emotion_detection
from eyes_color_detection import f_eye_color
from beard_detection import f_get_beard
from hair_type_detection import f_get_hair_type




#instantiate detectors
age_detector = f_my_age.Age_Model()
gender_detector =  f_my_gender.Gender_Model()
race_detector = f_my_race.Race_Model()
emotion_detector = f_emotion_detection.predict_emotions()

#----------------------------------------------



def get_face_info(im):
    # face detection
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            #face segmentation
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "eyes":[],
                "age":[],
                "gender":[],
                "race":[],
                "emotion":[],
                "beard":[],
                "hair":[],
                "bbx_frontal_face":box_face             
            } 

            face_image = im[x0:x1,y0:y1]

            #---------------------------------------eyes_detection--------------------------------------
            
            eye_detector = f_eye_color.eye_color(face_image)
            face_features["eyes"] = eye_detector
            
            #---------------------------------------beard_detection--------------------------------------
            
            beard_detector = f_get_beard.get_beard(face_image)
            face_features["beard"] = beard_detector
            
             #---------------------------------------hair_detection--------------------------------------
            
            hair_detector = f_get_hair_type.get_hair_type(face_image)
            face_features["hair"] = hair_detector
            

            # -------------------------------------- age_detection ---------------------------------------
            age = age_detector.predict_age(face_image)
            face_features["age"] = str(round(age,2))

            # -------------------------------------- gender_detection ---------------------------------------
            face_features["gender"] = gender_detector.predict_gender(face_image)

            # -------------------------------------- race_detection ---------------------------------------
            face_features["race"] = race_detector.predict_race(face_image)

            # -------------------------------------- emotion_detection ---------------------------------------
            _,emotion = emotion_detector.get_emotion(im,[box_face])
            face_features["emotion"] = emotion[0]

            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            "eyes":[],
            "age":[],
            "gender":[],
            "race":[],
            "emotion":[],
            "beard":[],
            "hair":[],
            "bbx_frontal_face":[]             
        }
        out.append(face_features)
    return out



def bounding_box(out,img):
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue
        else:
            x0,y0,x1,y1 = box
            img = cv2.rectangle(img,
                            (x0,y0),
                            (x1,y1),
                            (0,255,0),2);
            thickness = 1
            fontSize = 0.5
            step = 13

            try:
                cv2.putText(img, "age: " +data_face["age"], (x0, y0-7), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "gender: " +data_face["gender"], (x0, y0-step-10*1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "race: " +data_face["race"], (x0, y0-step-10*2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "emotion: " +data_face["emotion"], (x0, y0-step-10*3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "eyes: " +data_face["eyes"], (x0, y0-step-10*4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "beard: " +data_face["beard"], (x0, y0-step-10*5), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "hair: " +data_face["hair"], (x0, y0-step-10*6), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
    return img

