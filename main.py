# Importing the necessary libraries
import numpy as np
import time
import cv2
from pygame import mixer


# This function plays the corresponding drum beat if a green color object is detected in the region
def play_beat(detected, sound):
    # Checks if the detected green color is greater that a preset value
    play = (detected) > hat_thickness[0] * hat_thickness[1] * 0.8

    # If it is detected play the corresponding drum beat
    if play and sound == 1:
        drum_snare.play()

    elif play and sound == 2:
        drum_hat.play()
        time.sleep(0.001)


# This function is used to check if green color is present in the small region
def detect_in_region(frame, sound):
    # Converting to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Creating mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    # Calculating the number of green pixels
    detected = np.sum(mask)

    # Call the function to play the drum beat
    play_beat(detected, sound)

    return mask

def play_beat_2(detected, sound):
    # Checks if the detected green color is greater that a preset value
    play = (detected) > img1_thickness[0] * img1_thickness[1] * 0.8

    # If it is detected play the corresponding drum beat
    if play and sound == 4:
        drum_bass.play()

    elif play and sound == 3:
        drum_floor_tom.play()
        time.sleep(0.001)


def detect_in_region_2(frame, sound):
    # Converting to HSV
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Creating mask
    mask2 = cv2.inRange(hsv2, greenLower, greenUpper)

    # Calculating the number of green pixels
    detected = np.sum(mask2)

    # Call the function to play the drum beat
    play_beat_2(detected, sound)

    return mask


# A flag variable to choose whether to show the region that is being detected
verbose = False

# Importing drum beats
mixer.init()
drum_hat = mixer.Sound('./sounds/high_hat_1.ogg')
drum_snare = mixer.Sound('./sounds/snare_1.wav')
drum_bass = mixer.Sound('./sounds/Bass-Drum-1.wav')
drum_floor_tom = mixer.Sound('./sounds/Floor-Tom-1.wav')


# Set HSV range for detecting green color
greenLower = (0, 70, 50)
greenUpper = (10, 255, 255)

# Obtain input from the webcam
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
H, W = frame.shape[:2]

kernel = np.ones((7, 7), np.uint8)

# Read the image of High Hat and the Snare drum
hat = cv2.resize(cv2.imread('./images/3.png'), (200, 100), interpolation=cv2.INTER_CUBIC)
snare = cv2.resize(cv2.imread('./images/4.png'), (200, 100), interpolation=cv2.INTER_CUBIC)

# Set the region area for detecting green color
hat_center = [np.shape(frame)[1] * 2 // 8, np.shape(frame)[0] * 6 // 7]
snare_center = [np.shape(frame)[1] * 6 // 8, np.shape(frame)[0] * 6 // 7]

hat_thickness = [200, 100]
hat_top = [hat_center[0] - hat_thickness[0] // 2, hat_center[1] - hat_thickness[1] // 2]
hat_btm = [hat_center[0] + hat_thickness[0] // 2, hat_center[1] + hat_thickness[1] // 2]

snare_thickness = [200, 100]
snare_top = [snare_center[0] - snare_thickness[0] // 2, snare_center[1] - snare_thickness[1] // 2]
snare_btm = [snare_center[0] + snare_thickness[0] // 2, snare_center[1] + snare_thickness[1] // 2]


img1 = cv2.resize(cv2.imread('./images/1.png'), (200, 100), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(cv2.imread('./images/2.png'), (200, 100), interpolation=cv2.INTER_CUBIC)

# Set the region area for detecting green color
img1_center = [np.shape(frame)[1] * 2 // 8, np.shape(frame)[0] * 6 // 28]
img2_center = [np.shape(frame)[1] * 6 // 8, np.shape(frame)[0] * 6 // 28]

img1_thickness = [200, 100]
img1_top = [img1_center[0] - img1_thickness[0] // 2, img1_center[1] - img1_thickness[1] // 2]
img1_btm = [img1_center[0] + img1_thickness[0] // 2, img1_center[1] + img1_thickness[1] // 2]

img2_thickness = [200, 100]
img2_top = [img2_center[0] - img2_thickness[0] // 2, img2_center[1] - img2_thickness[1] // 2]
img2_btm = [img2_center[0] + img2_thickness[0] // 2, img2_center[1] + img2_thickness[1] // 2]


time.sleep(1)

while True:

    # Select the current frame
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if not (ret):
        break

    # Select region corresponding to the Snare drum
    snare_region = np.copy(frame[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]])
    mask = detect_in_region(snare_region, 1)

    # Select region corresponding to the High Hat
    hat_region = np.copy(frame[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]])
    mask = detect_in_region(hat_region, 2)


    # Select region corresponding to the Floor Tom
    img2_region = np.copy(frame[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]])
    mask2 = detect_in_region_2(img2_region, 3)

    # Select region corresponding to the Bass Rrum
    img1_region = np.copy(frame[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]])
    mask2 = detect_in_region_2(img1_region, 4)

    y = frame.shape[0]
    x = frame.shape[1]


    # Output project title
    # cv2.putText(frame, 'Virtual Drum', (220, 30), 2, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'Virtual Drums', (220, 30), 2, 1, (242, 232, 9), 2)
    cv2.putText(frame, 'Press Esc to exit', (10, y - 5), 1, 1, (255, 0, 0), 1)
    cv2.putText(frame, 'Stick color: RED', (x - 150, y - 5), 1, 1, (0, 0, 255), 1)

    # If flag is selected, display the region under detection
    if verbose:
        frame[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]] = cv2.bitwise_and(
            frame[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]],
            frame[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]],
            mask=mask[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]])
        frame[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]] = cv2.bitwise_and(
            frame[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]], frame[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]],
            mask=mask[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]])


        frame[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]] = cv2.bitwise_and(
            frame[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]],
            frame[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]],
            mask=mask2[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]])
        frame[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]] = cv2.bitwise_and(
            frame[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]],
            frame[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]],
            mask=mask2[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]])


    # If flag is not selected, display the drums
    else:
        frame[snare_top[1]:snare_btm[1], snare_top[0]:snare_btm[0]] = cv2.addWeighted(snare, 1,
                                                                                      frame[snare_top[1]:snare_btm[1],
                                                                                      snare_top[0]:snare_btm[0]], 1, 0)
        frame[hat_top[1]:hat_btm[1], hat_top[0]:hat_btm[0]] = cv2.addWeighted(hat, 1, frame[hat_top[1]:hat_btm[1],
                                                                                      hat_top[0]:hat_btm[0]], 1, 0)


        frame[img2_top[1]:img2_btm[1], img2_top[0]:img2_btm[0]] = cv2.addWeighted(img2, 1,
                                                                                  frame[img2_top[1]:img2_btm[1],
                                                                                  img2_top[0]:img2_btm[0]], 1, 0)
        frame[img1_top[1]:img1_btm[1], img1_top[0]:img1_btm[0]] = cv2.addWeighted(img1, 1,
                                                                                  frame[img1_top[1]:img1_btm[1],
                                                                                  img1_top[0]:img1_btm[0]], 1, 0)

    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xFF
    # 'Esc' to exit
    if key == 27:
        break

# Clean up the open windows
camera.release()
cv2.destroyAllWindows()