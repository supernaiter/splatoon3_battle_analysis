import datetime
from collections import deque
import csv
import random
import glob
import torch
import numpy as np
import cv2
import statistics
import os

from PIL import Image
import time

import re
import platform
if platform.system() == 'Darwin':  # Mac OS
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
# Image
img = "../battle.png"

# Inference
model = torch.hub.load('.', 'custom', path='../models/the_model.pt', source='local')
# model.conf = 0.25  # NMS confidence threshold
model.iou = 0.3  # NMS IoU threshold

ocr_model = torch.hub.load('.', 'custom', path='../models/ocr_model.pt', source='local')
message_ocr_model = torch.hub.load('.', 'custom', path='../models/message_ocr_model.pt', source='local')


results = model(img, 640)
results.print()
results_names = results.names

moving_count_class_num = [k for k, v in results_names.items() if v == 'moving_count'][0]
fixed_count_class_num = [k for k, v in results_names.items() if v == 'fixed_count'][0]
penalty_class_num = [k for k, v in results_names.items() if v == 'penalty'][0]

message_num = [k for k, v in results_names.items() if v == 'message'][0]
alive_num = [k for k, v in results_names.items() if v == 'alive'][0]
dead_num = [k for k, v in results_names.items() if v == 'dead'][0]
# special_num = [k for k, v in results_names.items() if v == 'special'][0]

hoko_kanmon_num = [k for k, v in results_names.items() if v == 'hoko_canmon'][0]
yagura_kanmon_num = [k for k, v in results_names.items() if v == 'yagura_kanmon'][0]
area_object_num = [k for k, v in results_names.items() if v == 'area_object'][0]
asari_object_num = [k for k, v in results_names.items() if v == 'asari_object'][0]
player_num = [k for k, v in results_names.items() if v == 'player'][0]


def output_ikalump_array(results):
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    alive_array = np_results[np_results[:, 5] == alive_num]

    dead_array = np_results[np_results[:, 5] == dead_num]

    special_array = np_results[np_results[:, 5] == special_num]

    ikalump_array = np.concatenate([alive_array, dead_array, special_array])
    # ikalump_array = np.concatenate([alive_array, dead_array])
    ikalump_array = ikalump_array[np.argsort(ikalump_array[:, col_num])]
    return ikalump_array


def output_alive_array(results):
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    alive_array = np_results[np_results[:, 5] == alive_num]
    return alive_array



def output_weapons_images(results):
    ikalump_array = output_ikalump_array(results)
    if len(ikalump_array) == 8:
        imgs = []
        img = results.ims[0]
        for i in range(0, 8):
            imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),
                        int(ikalump_array[i][0]):int(ikalump_array[i][2])])
        return imgs
    else:
        # print("not enough ikalumps.")
        return None


def output_weapons_images_for_special(results):
    ikalump_array = output_ikalump_array(results)
    imgs = []
    img = results.ims[0]
    for i in range(0, 8):
        imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),
                    int(ikalump_array[i][0]):int(ikalump_array[i][2])])
    return imgs


def output_ikalump_line(results):
    np_results = results.xyxy[0].cpu().numpy()
    ikalump_array = output_ikalump_array(results)
    ikalump_line = ikalump_array[:, 5]
    return ikalump_line


def output_count_array(results):
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    moving_count_array = np_results[np_results[:, 5] == moving_count_class_num]
    fixed_count_array = np_results[np_results[:, 5] == fixed_count_class_num]
    count_array = np.concatenate([moving_count_array, fixed_count_array])
    count_array = count_array[np.argsort(count_array[:, col_num])]
    return count_array


def get_center(results):
    np_results = results.xyxy[0].cpu().numpy()
    col_num = 0
    alive_array = np_results[np_results[:, 5] == alive_num]
    dead_array = np_results[np_results[:, 5] == dead_num]
    ikalump_array = np.concatenate([alive_array, dead_array])
    # print(len(ikalump_array))

    if len(ikalump_array) == 8:
        a = np.sum(ikalump_array[:, 0])
        b = np.sum(ikalump_array[:, 2])
        sum = a + b

        return sum // 16

    else:
        # print("ikalump_array number is not 8.")
        return results.ims[0].shape[1] // 2



#%%
f = open("../main_weapon_list.txt","r")
main_list = []
for x in f:
    main_list.append(x.rstrip("\n"))
    #以下のようにしてしまうと、改行コードがlistに入ってしまうため注意
    #list_row.append(x)
f.close()
print(len(main_list))
#%%
from torchvision import transforms
import torch.nn.functional as F

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)
    
mean = (0.5,)
std = (0.5,)

transform = ImageTransform(mean, std)

#weapon_model = torch.load('../230206_main_weapons_classification_weight.pth', map_location=device)
weapon_model = torch.load('../main_weapons_classification_weight.pth', map_location=device)
weapon_model.eval()  ## torch.nn.Module.eval



def output_weapon_names_pytorch(results, weapon_model, main_list):
    with torch.no_grad():
        weapon_classification_list = []
        #img = Image.open("52-Gal.jpg")

        imgs = output_weapons_images(results)
        for img in imgs:
            img = Image.fromarray(img)
            inputs = transform(img)
            inputs = inputs.unsqueeze(0).to(device)
            outputs = weapon_model(inputs)
            batch_probs = F.softmax(outputs, dim=1)
            batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
            for probs, indices in zip(batch_probs, batch_indices):
                weapon_classification_list.append(main_list[indices[0]])
    return weapon_classification_list

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def OCR_number(img, client):
    content = cv2.imencode(".png", img)[1].tobytes()
    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.document_text_detection(
        image=image,
        image_context={'language_hints': ['ja']}
    )

    # レスポンスからテキストデータを抽出
    output_text = ''
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    output_text += ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                output_text += '\n'
    return output_text


def output_count_numbers(results):
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    center = get_center(results)
    col_num = 0
    moving_count_array = np_results[np_results[:, 5] == moving_count_class_num]
    fixed_count_array = np_results[np_results[:, 5] == fixed_count_class_num]
    count_array = np.concatenate([moving_count_array, fixed_count_array])
    count_array = count_array[np.argsort(count_array[:, col_num])]
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                                 int(count_array[i][0]):int(count_array[i][2])]

            OCR_value = ocr_using_yolo(ocr_model, img)
            try:
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if (count_array[i][0] + count_array[i][2]) / 2 < center:
                    if 0 < OCR_result < 100:
                        count_list[0] = OCR_result

                elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                    if 0 < OCR_result < 100:
                        count_list[1] = OCR_result

            except:
                pass
            # imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),int(ikalump_array[i][0]):int(ikalump_array[i][2])])

    return count_list


def output_penalty_numbers(results):
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    center = get_center(results)
    col_num = 0
    count_array = np_results[np_results[:, 5] == penalty_class_num]
    count_array = count_array[np.argsort(count_array[:, col_num])]
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                                 int(count_array[i][0]):int(count_array[i][2])]

            try:
                OCR_value = ocr_using_yolo(ocr_model, img)
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if 0 < OCR_result < 100:
                    if (count_array[i][0] + count_array[i][2]) / 2 < center:
                        count_list[0] = OCR_result
                    elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                        count_list[1] = OCR_result
            except:
                pass

    return count_list

# detect_last_message


def get_image(results, num):
    return_array = None
    np_results = results.xyxy[0].cpu().numpy()
    array = np_results[np_results[:, 5] == num]
    if len(array) > 0:
        img = results.ims[0][int(array[0][1]):int(array[0][3]), int(array[0][0]):int(array[0][2])]
        return_array = img
    return return_array



def pytorch_weapon_classification(warm_up_batch):
    raw_outputs = []
    final_weapon_result = []
    for batch_result in warm_up_batch:
        raw_outputs.append(output_weapon_names_pytorch(results, weapon_model, main_list))
    
    raw_outputs = np.array(raw_outputs)
    for i in range(raw_outputs.shape[1]):
        final_weapon_result.append(statistics.mode(raw_outputs[:,i]))

    return final_weapon_result

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height),
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def ocr_using_yolo(ocr_model, count_img):
    ocr_results = ocr_model(count_img, 64)
    # ocr_results.print()
    # ocr_results = ocr_model(count_img, 128)
    numbers_list = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nokori']
    output = ""
    col_num = 0
    np_ocr_results = ocr_results.xyxy[0].cpu().numpy()
    ocr_array = np_ocr_results[np_ocr_results[:, 5] < 11]
    ocr_array = ocr_array[np.argsort(ocr_array[:, col_num])]
    # print(ocr_array)

    for arr in ocr_array:
        # print(str(numbers_list[int(arr[5])]))
        output = output + str(numbers_list[int(arr[5])])

    return output


def output_count_numbers(results):
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    center = get_center(results)
    col_num = 0
    moving_count_array = np_results[np_results[:, 5] == moving_count_class_num]
    fixed_count_array = np_results[np_results[:, 5] == fixed_count_class_num]
    count_array = np.concatenate([moving_count_array, fixed_count_array])
    count_array = count_array[np.argsort(count_array[:, col_num])]
    # print(count_array)
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                                 int(count_array[i][0]):int(count_array[i][2])]


            OCR_value = ocr_using_yolo(ocr_model, img)
            try:
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if (count_array[i][0] + count_array[i][2]) / 2 < center:
                    if 0 < OCR_result < 101:
                        count_list[0] = OCR_result

                elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                    if 0 < OCR_result < 101:
                        count_list[1] = OCR_result

            except:
                pass
            # imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),int(ikalump_array[i][0]):int(ikalump_array[i][2])])

    return count_list


def output_penalty_numbers(results):
    count_list = [None, None]
    np_results = results.xyxy[0].cpu().numpy()
    center = get_center(results)
    col_num = 0
    count_array = np_results[np_results[:, 5] == penalty_class_num]
    count_array = count_array[np.argsort(count_array[:, col_num])]
    if len(count_array) > 0:
        for i in range(len(count_array)):
            img = results.ims[0][int(count_array[i][1]):int(count_array[i][3]),
                                 int(count_array[i][0]):int(count_array[i][2])]

            try:
                OCR_value = ocr_using_yolo(ocr_model, img)
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if 0 < OCR_result < 100:
                    if (count_array[i][0] + count_array[i][2]) / 2 < center:
                        count_list[0] = OCR_result
                    elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                        count_list[1] = OCR_result
            except:
                pass
                # imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),int(ikalump_array[i][0]):int(ikalump_array[i][2])])

    return count_list


message_dict = {}
message_dict[0] = "1"
message_dict[1] = "2"
message_dict[2] = "3"
message_dict[3] = "ア"
message_dict[4] = "ば"
message_dict[5] = "バ"
message_dict[6] = "チ"
message_dict[7] = "ちゅう"
message_dict[8] = "だ"
message_dict[9] = "第"
message_dict[10] = "ど"
message_dict[11] = "ド"
message_dict[12] = "エ"
message_dict[13] = "防"
message_dict[14] = "が"
message_dict[15] = "ガ"
message_dict[16] = "グ"
message_dict[17] = "保"
message_dict[18] = "ホ"
message_dict[19] = "い"
message_dict[20] = "カ"
message_dict[21] = "確"
message_dict[22] = "け"
message_dict[23] = "こ"
message_dict[24] = "コ"
message_dict[25] = "みな"
message_dict[26] = "も"
message_dict[27] = "モ"
message_dict[28] = "ン"
message_dict[29] = "に"
message_dict[30] = "お"
message_dict[31] = "おう"
message_dict[32] = "破"
message_dict[33] = "プ"
message_dict[34] = "ラ"
message_dict[35] = "れ"
message_dict[36] = "リ"
message_dict[37] = "る"
message_dict[38] = "さ"
message_dict[39] = "し"
message_dict[40] = "ス"
message_dict[41] = "スー"
message_dict[42] = "た"
message_dict[43] = "ト"
message_dict[44] = "突"
message_dict[45] = "到"
message_dict[46] = "っ"
message_dict[47] = "ツ"

message_dict[48] = "着"
message_dict[49] = "う"
message_dict[50] = "ウ"
message_dict[51] = "失"


message_dict[52] = "わ"
message_dict[53] = "を"
message_dict[54] = "ヲ"
message_dict[55] = "ヤ"


def ocr_message_yolo(message_ocr_model, message_img):
    ocr_results = message_ocr_model(message_img, 640)
    # ocr_results.print()
    # ocr_results = ocr_model(count_img, 128)
    output = ""
    col_num = 0
    np_ocr_results = ocr_results.xyxy[0].cpu().numpy()
    ocr_array = np_ocr_results
    ocr_array = ocr_array[np.argsort(ocr_array[:, col_num])]

    for arr in ocr_array:
        output = output + message_dict[arr[5]]

    return output

#%% settings
img = "../battle.png"

results = model(img, 640)
results.print()
results_names = results.names

moving_count_class_num = [k for k, v in results_names.items() if v == 'moving_count'][0]
fixed_count_class_num = [k for k, v in results_names.items() if v == 'fixed_count'][0]
penalty_class_num = [k for k, v in results_names.items() if v == 'penalty'][0]

message_num = [k for k, v in results_names.items() if v == 'message'][0]
alive_num = [k for k, v in results_names.items() if v == 'alive'][0]
dead_num = [k for k, v in results_names.items() if v == 'dead'][0]
special_num = [k for k, v in results_names.items() if v == 'special'][0]

hoko_kanmon_num = [k for k, v in results_names.items() if v == 'hoko_canmon'][0]
yagura_kanmon_num = [k for k, v in results_names.items() if v == 'yagura_kanmon'][0]
area_object_num = [k for k, v in results_names.items() if v == 'area_object'][0]
asari_object_num = [k for k, v in results_names.items() if v == 'asari_object'][0]


#%% testing_using_sample image
img = "../sample_starting.png"

results = model(img, 640)
results.print()
results_names = results.names
output_weapon_names_pytorch(results, weapon_model, main_list)

#%%
l = glob.glob('../footages/*.mp4')
print(len(l))
random.shuffle(l)

#%%


csv_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
input_video_path = l[0]
cap = cv2.VideoCapture(input_video_path)
fps = 7
frame_interval = cap.get(cv2.CAP_PROP_FPS) // fps
print("fps", fps)
fps_original = cap.get(cv2.CAP_PROP_FPS)
count = 0
frame_count = 0
analysis_date = datetime.datetime.now()
# count_list = [100,100]
# penalty_list = [0,0]

# hundred_results = []
warm_up_results = []

# special_batch = []
warm_up_batch = []

# message_batch = []

# final_result = []
final_result = deque()

start_frame = None

in_game = False
weapon_list = []
detected_rule = None
detected_stage = None

ikalump_combo = 0
no_signal_combo = 0
match_count = 0
state = 0
# fps = 1
prep_flag = True
warm_up_frames = 10

start_timestamp = None

saved_center = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2

while True:
    count += 1
    if count % frame_interval == 0:
        ret, frame = cap.read()
        startingts = time.time()
        if not ret:
            print('break')
            break
        frame_count += 1
        
        # Create a copy of frame for display
        display_frame = frame.copy()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        before = time.time()
        results = model(frame, 640)
        np_results = results.xyxy[0].cpu().numpy()
        # print(np_results)
        alive_count = np.sum(np_results[:, 5] == alive_num)
        dead_count = np.sum(np_results[:, 5] == dead_num)
        special_count = np.sum(np_results[:, 5] == special_num)
        result_list = [None] * 33

        area_count = np.sum(np_results[:, 5] == area_object_num)
        asari_count = np.sum(np_results[:, 5] == asari_object_num)
        hoko_count = np.sum(np_results[:, 5] == hoko_kanmon_num)
        yagura_count = np.sum(np_results[:, 5] == yagura_kanmon_num)
        player_count = np.sum(np_results[:, 5] == player_num)

        map_info_count = np.sum(np_results[:, 5] == 8)
        map_player_dead_count = np.sum(np_results[:, 5] == 9)
        map_player_position_count = np.sum(np_results[:, 5] == 10)

        message_count = np.sum(np_results[:, 5] == message_num)

        all_count = alive_count + dead_count + special_count
        # print(alive_count, dead_count, special_count, all_count)
        map_count = map_info_count + map_player_dead_count + map_player_position_count

        result_list[29] = analysis_date
        result_list[28] = input_video_path

        if start_frame is not None:

            result_list[0] = round((frame_count - start_frame) * (1 / fps), 1)

        # Draw detection results on display frame
        for det in np_results:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"Class {int(cls)} ({conf:.2f})"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # rule_count = area_count + asari_count + hoko_count + nawabari_count + yagura_count
        if all_count == 8:
            if start_frame == None:
                start_frame = frame_count

            ikalump_state = list(output_ikalump_line(results))
            for i in range(len(ikalump_state)):
                result_list[1 + i] = ikalump_state[i]

            if warm_up_frames > 0:
                warm_up_batch.append(results)
                # weapon_list = output_weapon_names(results,weapon_model, main_list)
                # for i in range(len(weapon_list)):
                #    result_list[24+i] = weapon_list[i]

                # detected_stage = classify_stage(results, stage_model, stage_list)
                # result_list[33] = detected_stage

                warm_up_frames -= 1
                if warm_up_frames == 0:
                    # print("warm_up_frames",len(warm_up_batch))

                    # warm_up_resultsを処理する
                    #detected_stage = batch_stage_classification(warm_up_batch)
                    detected_stage = None
                    #weapon_list = batch_weapon_classification(warm_up_batch)
                    weapon_list = pytorch_weapon_classification(warm_up_batch)
                    print(result_list)
            # detected_rule = "yagura"
            if warm_up_frames == 0:
                result_list[21] = detected_stage
                for i in range(len(weapon_list)):
                    result_list[13 + i] = weapon_list[i]
            print("warm_up_ends.")

        result_list[22] = asari_count
        result_list[23] = hoko_count
        result_list[24] = area_count
        result_list[25] = yagura_count

        if message_count > 0:
            message = ocr_message_yolo(message_ocr_model, results.ims[0])
            # plt.imshow(results.ims[0])
            # plt.show()
            print(message)
            result_list[26] = message

        if player_count > 0:
            result_list[27] = True
        # print("second", time.time() - before)

        # count_list = output_count_numbers_pytesseract(results)
        # if frame_count % 5 == 0:
        count_list = output_count_numbers(results)
        for i in range(len(count_list)):
            result_list[9 + i] = count_list[i]
        # penalty_list = output_penalty_numbers_pytesseract(results)
        penalty_list = output_penalty_numbers(results)
        for i in range(len(penalty_list)):
            result_list[11 + i] = penalty_list[i]

        final_result.append(result_list)
        # print("last", time.time() - startingts)
        # Display frame with detections
        cv2.imshow("yolo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        ret = cap.grab()

final_result = list(final_result)

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    for res in final_result:
        writer.writerow(res)
# %%




