import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import csv

user_object = tf.saved_model.load('models/') # 加载 SavedModel
model = user_object.signatures['serving_default'] # 获取模型对象
print('*'*50)
print(model)
print('*'*50)
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

vdu_path = 'hands.mp4'
vdu = cv2.VideoCapture(vdu_path)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

vdu_seq = []
seq_length = 0
threshold = 100

def resize_image(img, width=None, height=None):
    dim = (width, height) if width and height else (800, 600)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

def window_adjust(name, width, height, x_pos, y_pos, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.moveWindow(name, x_pos, y_pos)
    cv2.imshow(name, img)

while vdu.isOpened():
    ret, img = vdu.read()

    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            print("*" * 50)
            print(res)
            print("res.landmark = ".format(res.landmark))
            print("*" * 50)
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            vdu_d = np.concatenate([joint.flatten(), angle])

            vdu_seq.append(vdu_d)
            seq_length += 1

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) == ord('q'):
        break

vdu.release()
cv2.destroyAllWindows()

file_path = 'vdu_seq.csv'

with open(file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)

    for gesture_data in vdu_seq:
        if not isinstance(gesture_data, list):
            gesture_data = gesture_data.tolist()

        csv_writer.writerow(gesture_data)

print(f"Gesture data saved to {file_path}")

vdu_seq = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        gesture_data = [float(value) for value in row]

        vdu_seq.append(np.array(gesture_data))

index = -1

match_time = 0
miss_time = 0
last_feedback_time = time.time()

vdu = cv2.VideoCapture(vdu_path)
cap = cv2.VideoCapture(0)

def feedback(match_time, miss_time):
    if(2 * match_time > miss_time):
        print("Good!, keep that way!")
    else:
        print("Try to catch up!")

def gesture_similar(video_seq, index, cap_d, threshold, img, match_time, miss_time):
    video_d = video_seq[index]
    distance = np.linalg.norm(video_d - cap_d)

    print_text = ''
    if distance < threshold:
        match_time += 1
        print_text = 'match'
    else:
        miss_time += 1
        print_text = 'miss'
    
    cv2.putText(img, print_text, org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    return match_time, miss_time

while vdu.isOpened() and cap.isOpened():
    ret, img = cap.read()
    ret, smp = vdu.read()

    if not ret:
        break

    index = (index + 1) % seq_length
    
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            cap_d = np.concatenate([joint.flatten(), angle])

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            match_time, miss_time = gesture_similar(vdu_seq, index, cap_d, threshold, img, match_time, miss_time)

            if (time.time() - last_feedback_time) >= 5:
                feedback(match_time, miss_time)
                last_feedback_time = time.time()
                print("match_time = {}, miss_time = {}".format(match_time, miss_time))
                match_time = 0
                miss_time = 0

    smp_resized = resize_image(smp, width=800, height=600)
    img_resized = resize_image(img, width=800, height=600)

    window_adjust('video', width = 800, height = 600, x_pos = 0, y_pos = 0, img = smp_resized)
    window_adjust('camera', width = 800, height = 600, x_pos = 800, y_pos = 0, img = img_resized)

    if cv2.waitKey(1) == ord('q'):
        break

vdu.release()
cap.release()
cv2.destroyAllWindows()