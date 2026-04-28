import cv2
import time
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound

def calc_angle(p1, p2, p3):
    ang1 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    ang2 = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    res = math.degrees(ang1 - ang2)
    if res < 0:
        res += 360
    if res > 180:
        res = 360 - res
    return res

def check_rep(img, pts, state, total):
    l_sh, r_sh = pts[5], pts[6]
    l_el, r_el = pts[7], pts[8]
    l_wr, r_wr = pts[9], pts[10]
    
    beep = False

    if l_sh[0] > 0 and r_sh[0] > 0:
        ang_l = calc_angle(l_sh, l_el, l_wr)
        ang_r = calc_angle(r_sh, r_el, r_wr)

        if ang_l < 100 and ang_r < 100 and not state:
            state = True
        
        if ang_l > 150 and ang_r > 150 and state:
            state = False
            total += 1
            beep = True
            
            print("pushup")
        cv2.putText(img, f"Total: {total}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    return state, total, beep

nn = YOLO("yolo26n-pose.pt")
nn.to("mps")

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: 
    fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

audio_task = None
track = "sound.mp3" 

bottom = False
counter = 0
timer = time.time()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    if time.time() - timer > 5.0:
        counter = 0
        bottom = False
        
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

    start_t = time.perf_counter()
    preds = nn.predict(img, verbose=False)
    

    if preds:
        res = preds[0]
        pts = res.keypoints.xy.tolist()
        
        if pts and len(pts[0]) >= 11:
            timer = time.time()
            
            drawer = Annotator(img)
            drawer.kpts(res.keypoints.data[0], res.orig_shape, 5, True)
            drawn_img = drawer.result()
            
            bottom, counter, beep = check_rep(drawn_img, pts[0], bottom, counter)
            
            if beep:
                if audio_task is None or not audio_task.is_alive():
                    audio_task = playsound(track, block=False)
                    
            out_video.write(drawn_img)
            cv2.imshow("Tracker", drawn_img)
            continue

    out_video.write(img)
    cv2.imshow("Tracker", img)

cap.release()
out_video.release()
cv2.destroyAllWindows()