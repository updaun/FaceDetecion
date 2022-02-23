from FaceDetectionModule import FaceDetector

import cv2
import mediapipe as mp
import time

def draw_crop(img, face_lm, size, color=(255,255,255), thickness=2):
    lm_x, lm_y = face_lm
    img = cv2.rectangle(img, (lm_x-size, lm_y-size), (lm_x+size, lm_y+size), color, thickness)
    return img

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    detector = FaceDetector(minDetectionCon=0.5)

    while True:
        success, img = cap.read()

        ih, iw, ic = img.shape

        img, bbox, result = detector.findFacesWithResult(img, draw=False)

        if len(bbox) != 0:
            
            faces = []

            for id, lm in enumerate(result.detections[0].location_data.relative_keypoints):
                faces.append([int(lm.x * iw), int(lm.y*ih)])

            # print(faces)
            
            # left eye
            img = draw_crop(img, faces[1], 25, (200, 0,200))

            # right eye
            img = draw_crop(img, faces[0], 25, (200, 0,200))
            
            # nose
            img = draw_crop(img, faces[2], 25, (200, 0,0))

            # mouth
            img = draw_crop(img, faces[3], 25, (200, 200,0))


            cv2.imshow("face", img)

        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()