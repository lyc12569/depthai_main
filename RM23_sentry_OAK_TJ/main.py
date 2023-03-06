import cv2
import numpy as np
import depthai as dai
import time

from modules.OAK import OAKCap
from modules.detection import Detector
from modules.communication import Communicator
from modules.utilities import drawContour, drawPoint, drawAxis, putText

def zimiao(frame):
    start = time.time()
    lightBars, armors = detector.detect(frame)
    if len(armors) > 0:
        a = armors[0]  # TODO a = classifior.classify(armors)

        a.targeted(objPoints, cameraMatrix, distCoeffs)

        # TODO yaw, pitch = predictor.predict(a)

        if debug:
            drawAxis(frame, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            putText(frame, f'{a.yaw:.2f} {a.pitch:.2f}', a.center)
            drawPoint(frame, a.center, (255, 255, 255))

        if useSerial:
            communicator.send(a.yaw, -a.pitch * 0.5)
    else:
        if useSerial:
            communicator.send(0, 0)

    processTimeMs = (time.time() - start) * 1000
    print(f'{processTimeMs=}')

    if debug:
        for l in lightBars:
            drawContour(frame, l.points, (0, 255, 255), 10)
        for a in armors:
            drawContour(frame, a.points)
        cv2.convertScaleAbs(frame, frame, alpha=5)
        cv2.imshow('result', frame)
        output.write(frame)
        
def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# TODO config.toml
debug = True
useCamera = True
exposureMs = 0.5
useSerial = False
# port = '/dev/tty.usbserial-A50285BI'  # for ubuntu: '/dev/ttyUSB0'
port = '/dev/ttyUSB0'
cameraMatrix = np.array([[1025.35323576971, 0, 678.7266096913569],
 [0, 1027.080747783651, 396.9785232682882],
 [0, 0, 1]],   dtype=np.double)
distCoeffs = np.array([[0.220473935504143, -1.294976334542626, 0.003407354582097702, 
                        -0.001096689035107743, 2.91864887650898]], dtype=np.double)

expTime = 20000
sensIso = 800
wbManual = 4000
EXP_STEP = 500  # us
ISO_STEP = 50 
WB_STEP = 200

# TODO 大装甲板
lightBarLength, armorWidth = 56, 135
objPoints = np.float32([[-armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, lightBarLength / 2, 0],
                        [-armorWidth / 2, lightBarLength / 2, 0]])

detector = Detector()
if useSerial:
    communicator = Communicator(port)
if debug:
    output = cv2.VideoWriter('assets/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 1024))
    
if useCamera:
    cap=OAKCap() 
    pipeline,device=cap.startRgb()
    stereo = cap.startDisparity(pipeline)
    with device:
        device.startPipeline(pipeline)
        controlQueue = device.getInputQueue('control')
        while True:
            frame = cap.read(device)
            # frameDisp = cap.readDisp(device,stereo)
            zimiao(frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
                if key == ord('i'):
                    expTime -= EXP_STEP
                if key == ord('o'):
                    expTime += EXP_STEP
                if key == ord('k'):
                    sensIso -= ISO_STEP
                if key == ord('l'):
                    sensIso += ISO_STEP
                expTime = clamp(expTime, 1, 33000)
                sensIso = clamp(sensIso, 100, 1600)
                print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
                ctrl = dai.CameraControl()
                ctrl.setManualExposure(expTime, sensIso)
                controlQueue.send(ctrl)
            elif key in [ord('n'), ord('m')]:
                if key == ord('n'):
                    wbManual -= WB_STEP
                if key == ord('m'):
                    wbManual += WB_STEP
                wbManual = clamp(wbManual, 1000, 12000)
                print("Setting manual white balance, temperature: ", wbManual, "K")
                ctrl = dai.CameraControl()
                ctrl.setManualWhiteBalance(wbManual)
                controlQueue.send(ctrl)

            
else :
    cap = cv2.VideoCapture('assets/input.avi')
# detector = Detector()
# if useSerial:
#     communicator = Communicator(port)
# if debug:
#     output = cv2.VideoWriter('assets/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 1024))

    while True:
        success, frame = cap.read()
        if not success:
            break
        # frameDisp = cap.readDisp(device,stereo)
        zimiao(frame)
        # cv2.imshow('rgb',frameRgb)
        # cv2.imshow('disp',frameDisp)
        key = cv2.waitKey(1)
        if key == ord('q'):
                break
        



