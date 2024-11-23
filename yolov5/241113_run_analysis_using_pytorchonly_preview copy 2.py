
#%%
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

#import tensorflow as tf
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
"""
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

#%%

#ここのkeras部分を全部pytorchに変えたい
"""
weapon_model = tf.keras.models.load_model('models/weapon_keras_model')
main_list = ['52-Gal', '96-Gal', 'Aerospray', 'Ballpoint-Splatling', 'Bamboozler-14-Mk-I', 'Blaster', 'Bloblobber', 'Carbon-Roller', 'Clash-Blaster', 'Dapple-Dualies', 'Dark-Tetra-Dualies', 'Dualie-Squelchers', 'Dynamo-Roller', 'E-liter-4K', 'Explosher', 'Flingza-Roller', 'Glooga-Dualies', 'Goo-Tuber', 'H-3-Nozzlenose', 'Heavy-Splatling', 'Hero-Shooter-Replica', 'Hydra-Splatling', 'Inkbrush', 'Jet-Squelcher', 'L-3-Nozzlenose', 'LACT-450',
             'Luna-Blaster', 'Mini-Splatling', 'N-ZAP85', 'Nautilus-47', 'Octobrush', 'Range-Blaster', 'Rapid-Blaster', 'Rapid-Blaster-Pro', 'Slosher', 'Sloshing-Machine', 'Splash-o-matic', 'Splat-Brella', 'Splat-Charger', 'Splat-Dualies', 'Splat-Roller', 'Splatana-Stamper', 'Splatana-Wiper', 'Splattershot', 'Splattershot-Jr', 'Splattershot-Pro', 'Sploosh-o-matic', 'Squeezer', 'Squiffer', 'Tenta-Brella', 'Tri-Slosher', 'Tri-Stringer', 'Undercover-Brella']

stage_model = tf.keras.models.load_model('models/stage_keras_model')
stage_list = ['amabi', 'chozame', 'gonzui', 'kinmedai', 'mahimahi',
              'masaba', 'mategai', 'namerou', 'sumeshi', 'yagara', 'yunohana', 'zatou']
"""



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


def classify_stage(results, stage_model, stage_list):

    # im = tf.image.resize(cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB), (160,160))[tf.newaxis, ...]
    im = tf.image.resize(results.ims[0], (160, 160))[tf.newaxis, ...]
    # print(im.shape)
    # img_single = resize(im, IMAGE_SIZE)
    imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(im)

    # predict =  stage_model.predict(imgs_preprocessed)
    predict = stage_model(imgs_preprocessed).numpy()
    print(list(predict))
    return stage_list[predict.argmax()]


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


def output_weapon_names(results, weapon_model, main_list):
    weapon_classification_list = []
    imgs = output_weapons_images(results)
    imgs_np = []
    if imgs is not None:
        for im in imgs:
            img_np = tf.image.resize(im, (160, 160))
            # print(im.shape)
            imgs_np.append(img_np)
        imgs_np = np.array(imgs_np)
        imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)
        predict = weapon_model.predict(np.array(imgs_preprocessed))
        for p in predict:
            weapon_classification_list.append(main_list[p.argmax()])
    return weapon_classification_list

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

weapon_model = torch.load('../230206_main_weapons_classification_weight.pth',map_location=device, weights_only=False)
weapon_model.eval()  ## torch.nn.Module.eval


#%%
"""
img = Image.open("../52-Gal.jpg")

inputs = transform(img)
print(inputs.shape)
inputs = inputs.unsqueeze(0).to(device)
print(inputs.shape)
model = torch.load('../main_weapons_classification_weight.pth')
model.eval()  ## torch.nn.Module.eval

with torch.no_grad():
    outputs = model(inputs)
    print(outputs.shape)
    batch_probs = F.softmax(outputs, dim=1)
    batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
    print(batch_indices)
    print(batch_indices.shape)
    for probs, indices in zip(batch_probs, batch_indices):
        for k in range(1):
            print(k)
            print(indices[k])
            print(main_list[indices[k]])
"""
#%%
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

#%%


# カウントをOCRする．

# import pytesseract
def pil_image_to_base64(np_image):
    pil_image = Image.fromarray(np_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    str_encode_file = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return str_encode_file


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def OCR_number_pytesseract(img):
    img = Image.fromarray(img)
    original_size = img.size

    input_image = crop_center(img, original_size[0], original_size[1] // 1.5)
    # input_image = img

    input_image.show()
    threshold = 150
    gray_image = input_image.convert("L")
    # gray.point(lambda x: 0 if x < thre else x)
    size = gray_image.size
    binary_image = Image.fromarray(((np.array(gray_image) < threshold) * 255).astype(np.uint8)
                                   ).resize((size[0] * 2, size[1] * 2)).convert("L").filter(ImageFilter.SMOOTH_MORE)
    binary_image.show()
    text = pytesseract.image_to_string(np.array(binary_image, dtype=np.uint8), lang="spl_num",
                                       config="--psm 8 -c tessedit_char_blacklist=+:").replace("\x0c", "").replace("\n", "")

    return text


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

            # pil_img = Image.fromarray(img)
            # pil_img.show()
            # vision_image = vision.Image(content=pil_img)
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # content = pil_image_to_base64(img)
            # ここで本来ならOCRで数字を取得する
            # OCR_value =OCR_number(img, client)
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

            # pil_img = Image.fromarray(img)
            # pil_img.show()
            # vision_image = vision.Image(content=pil_img)
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # content = pil_image_to_base64(img)
            # ここで本来ならOCRで数字を取得する
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

# detect_last_message


def get_image(results, num):
    return_array = None
    np_results = results.xyxy[0].cpu().numpy()
    array = np_results[np_results[:, 5] == num]
    if len(array) > 0:
        img = results.ims[0][int(array[0][1]):int(array[0][3]), int(array[0][0]):int(array[0][2])]
        return_array = img
    return return_array


def classify_message(results, message_model, message_list):
    classified_message = "no_message"
    img_ex = get_image(results, message_num)
    if img_ex is not None:
        im = tf.image.resize(img_ex, (160, 160))
        img_single = im[tf.newaxis, ...]
        imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_single)
        # predict =  message_model.predict(imgs_preprocessed)
        predict = list(message_model(imgs_preprocessed).numpy())
        classified_message = message_list[predict.argmax()]
    return classified_message


def detect_special(results, special_model):
    special_classification_list = []
    imgs = output_weapons_images_for_special(results)
    imgs_np = []
    if imgs is not None:
        for im in imgs:
            img_np = tf.image.resize(im, (64, 64))
            # print(im.shape)
            imgs_np.append(img_np)
        imgs_np = np.array(imgs_np)
        imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)
        # predict =  special_model.predict(np.array(imgs_preprocessed))
        predict = list(special_model(np.array(imgs_preprocessed)).numpy())
        for p in predict:
            special_classification_list.append(p.argmax())
    return special_classification_list


def batch_stage_classification(warm_up_batch):
    stage_classification_list = []
    imgs = []
    for result in warm_up_batch:
        img_ex = tf.image.resize(result.ims[0], (160, 160))
        imgs.append(img_ex)
    imgs_np = np.array(imgs)
    imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)

    predict = stage_model.predict_on_batch(imgs_preprocessed)

    for p in predict:
        stage_classification_list.append(stage_list[p.argmax()])
    return statistics.mode(stage_classification_list)


def batch_weapon_classification(warm_up_batch):
    final_weapon_result = []
    weapon_classification_list = []
    imgs = []
    for result in warm_up_batch:
        imgs.extend(output_weapons_images_for_special(result))
    """

    for img in imgs:
        imgs_np.append(tf.image.resize(img, (160, 160)))

    imgs_np = np.array(imgs_np)
    imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)

    predict = weapon_model.predict_on_batch(imgs_preprocessed)

    for p in predict:
        weapon_classification_list.append(p.argmax())
    
    """

    np_class_array = np.array(weapon_classification_list).reshape(len(weapon_classification_list) // 8, 8)
    print(np_class_array)
    for i in range(np_class_array.shape[1]):
        final_weapon_result.append(main_list[statistics.mode(np_class_array[:, i])])
    return final_weapon_result


def pytorch_weapon_classification(warm_up_batch):
    raw_outputs = []
    final_weapon_result = []
    for batch_result in warm_up_batch:
        raw_outputs.append(output_weapon_names_pytorch(results, weapon_model, main_list))
    
    raw_outputs = np.array(raw_outputs)
    for i in range(raw_outputs.shape[1]):
        final_weapon_result.append(statistics.mode(raw_outputs[:,i]))

    return final_weapon_result

def batch_special_classification(special_batch, hundred_results):
    special_classification_list = []
    imgs = []
    for result in special_batch:
        imgs.extend(output_weapons_images_for_special(result))
    imgs_np = []

    for img in imgs:
        # plt.imshow(img)
        # plt.show()
        imgs_np.append(tf.image.resize(img, (64, 64)))

    imgs_np = np.array(imgs_np)

    imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)

    predict = special_model.predict_on_batch(imgs_preprocessed)

    for p in predict:
        special_classification_list.append(p.argmax())

    # for arr in np_class_array:
    np_class_array = np.array(special_classification_list).reshape(len(special_classification_list) // 8, 8)
    for i, arr in enumerate(np_class_array):
        for j in range(len(arr)):
            hundred_results[i][9 + j] = arr[j]

    return hundred_results


def batch_message_classification(special_batch, hundred_results):
    message_classification_list = []
    batch_list = []
    imgs = []
    for result in special_batch:
        img_ex = get_image(result, message_num)
        if img_ex is not None:
            img = Image.fromarray(img_ex)
            # print(img.size)
            threshold = 230
            gray_image = img.convert("L")
            size = gray_image.size
            binary_image = Image.fromarray(((np.array(gray_image) > threshold) * 255).astype(np.uint8))

            # binary_image.show()

            new_image = Image.new("L", (size[0] + 100, size[0] + 100), 0)
            new_image.paste(binary_image, (50, size[0] // 2))

            # new_image.show()
            new_image = new_image.convert("RGB")
            new_image = np.asarray(new_image, np.uint8)
            im = tf.image.resize(new_image, (160, 160))
            imgs.append(im)
        else:
            imgs.append(np.zeros((160, 160, 3)))

    imgs_np = np.array(imgs)

    # print(imgs_np.shape)

    imgs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_np)

    predict = message_model.predict_on_batch(imgs_preprocessed)

    for p in predict:
        message_classification_list.append(p.argmax())
    # print(message_classification_list)

    for i, arr in enumerate(predict):
        hundred_results[i][21] = message_list[arr.argmax()]
        if arr.argmax() > 0:
            print(message_list[arr.argmax()])

    return hundred_results


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height),
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def OCR_number_pytesseract(img):
    gray_image = img
    text = None
    # input_image = img
    # input_image.show()
    # gray_image = input_image.convert("L")
    # gray.point(lambda x: 0 if x < thre else x)
    size = gray_image.size
    for threshold in range(160, 200, 10):
        # binary_image = Image.fromarray(((np.array(gray_image) < threshold)*255).astype(np.uint8)).resize((size[0]*2, size[1]*2)).convert("L").filter(ImageFilter.SMOOTH_MORE)
        binary_image = Image.fromarray(((np.array(gray_image) < threshold) * 255).astype(np.uint8)
                                       ).resize((size[0] * 2, size[1] * 2)).convert("L")
        # binary_image.show()
        text = pytesseract.image_to_string(np.array(binary_image, dtype=np.uint8), lang="spl_num",
                                           config="--psm 8 -c tessedit_char_blacklist=+:").replace("\x0c", "").replace("\n", "").replace(" ", "")
        # print(threshold,text)
        try:
            if 0 < int(text) < 101:
                break
        except Exception as E:
            pass
    return text


def output_count_numbers_pytesseract(results):
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

            # pil_img = Image.fromarray(img)
            # pil_img.show()
            # vision_image = vision.Image(content=pil_img)
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # content = pil_image_to_base64(img)
            # ここで本来ならOCRで数字を取得する
            img = Image.fromarray(img)
            original_size = img.size
            # img.show()
            img = crop_center(img, original_size[0], original_size[1] // 1.5)
            OCR_value = OCR_number_pytesseract(img)
            try:
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if (count_array[i][0] + count_array[i][2]) / 2 < center:
                    if 0 < OCR_result < 101:
                        count_list[0] = OCR_result

                elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                    if 0 < OCR_result < 101:
                        count_list[1] = OCR_result

            except Exception as E:
                print(E)
            # imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),int(ikalump_array[i][0]):int(ikalump_array[i][2])])

    return count_list


def output_penalty_numbers_pytesseract(results):
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

            # ここで本来ならOCRで数字を取得する
            img = Image.fromarray(img)
            original_size = img.size
            # img.show()
            img = img.crop(((original_size[0] // 2.5), 0, original_size[0], original_size[1]))

            try:
                OCR_value = OCR_number_pytesseract(img)
                OCR_result = int(re.sub(r"\D", "", OCR_value))
                if 0 < OCR_result < 101:
                    if (count_array[i][0] + count_array[i][2]) / 2 < center:
                        count_list[0] = OCR_result
                    elif (count_array[i][0] + count_array[i][2]) / 2 > center:
                        count_list[1] = OCR_result
            except:
                pass
                # imgs.append(img[int(ikalump_array[i][1]):int(ikalump_array[i][3]),int(ikalump_array[i][0]):int(ikalump_array[i][2])])

    return count_list


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

            # pil_img = Image.fromarray(img)
            # pil_img.show()
            # vision_image = vision.Image(content=pil_img)
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # content = pil_image_to_base64(img)
            # ここで本来ならOCRで数字を取得する
            # OCR_value =OCR_number(img, client)
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

            # pil_img = Image.fromarray(img)
            # pil_img.show()
            # vision_image = vision.Image(content=pil_img)
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # content = pil_image_to_base64(img)
            # ここで本来ならOCRで数字を取得する
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

for input_video_path in l:
    print(input_video_path)
    csv_path_ver_two = input_video_path.split(".")[0] + "_ver_two.csv"
    csv_path = input_video_path.split(".")[0] + "_ver_three.csv"
    if os.path.isfile(csv_path_ver_two):
        continue
    print(csv_path)
    cap = cv2.VideoCapture(input_video_path)
    fps = 10
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
            # cv2.imshow("yolo", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
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
                        # print(result_list)
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

        else:
            ret = cap.grab()

    final_result = list(final_result)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for res in final_result:
            writer.writerow(res)


if __name__ == "__main__":
    main()

# %%
