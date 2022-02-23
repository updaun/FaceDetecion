from cvzone.FaceMeshModule import FaceMeshDetector

import cv2
import mediapipe as mp
import time

def draw_crop(img, face_lm, size, color=(255,255,255), thickness=2):
    lm_x, lm_y = face_lm
    img = cv2.rectangle(img, (lm_x-size, lm_y-size), (lm_x+size, lm_y+size), color, thickness)
    return img

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    detector = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)

    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img, draw=False)
        # img, faces = detector.findFaceMesh(img, draw=True)

        if len(faces) != 0:
            
            # left eye
            img = draw_crop(img, faces[0][159], 25, (200, 0,200))

            # right eye
            img = draw_crop(img, faces[0][386], 25, (200, 0,200))
            
            # nose
            # img = cv2.circle(img, faces[0][4], 5, (255, 255, 255), 3)
            img = draw_crop(img, faces[0][4], 25, (200, 0,0))

            # mouth
            img = draw_crop(img, faces[0][14], 25, (200, 200,0))


            cv2.imshow("face", img)

        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()