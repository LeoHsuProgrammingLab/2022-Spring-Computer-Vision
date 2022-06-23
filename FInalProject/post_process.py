# import numpy as np

# def post_process(txt_path):
#   data = np.loadtxt(txt_path, dtype=np.float64)
#   data_len = data.shape[0]
#   result = np.zeros(shape=data.shape)
#   window_len = 5
#   copy_data = np.zeros(shape = data_len+2*(window_len//2))
#   for i in range(data_len):
#     copy_data[i+window_len//2] = data[i]
#   for i in range(window_len//2):
#     w_l = window_len//2 - i
#     copy_data[i] = copy_data[i+w_l*2]
#     copy_data[-i-1] = copy_data[(-i-1)-w_l*2]

#   data_len = copy_data.shape[0]

#   for i in range(window_len//2, data_len-window_len//2):
#     window = copy_data[i-window_len//2:i+window_len//2+1]
#     result[i-window_len//2] = np.median(window)

#   with open(txt_path, 'w') as f:
#     for item in result:
#       f.write(str(item)+"\n")

import numpy as np 
import os
from PIL import Image, ImageEnhance, ImageOps
import cv2
from tqdm.auto import tqdm

def get_avg(data_path, root):
  data_len = len(data_path)
  area = np.zeros(shape=data_len)
  for i in range(data_len):
    img = cv2.imread(root + str(i) + ".png")[:, :, 0]
    area[i] = np.sum(img>128) * 255
  avg_area = np.sum(area) / data_len
  for i in range(data_len):
    if area[i] < avg_area * 0.5:
      area[i] = 0
  return area, avg_area

def filter(area, avg_area):
  window_len = 5
  area_len = area.shape[0]
  result = np.zeros(shape = area_len)
  for i in range(window_len//2):
    if(area[i] > avg_area):
      result[i] = 1.0
    if(area[-i-1] > avg_area):
      result[-i-1] = 1.0
  for i in range(window_len // 2, area_len - window_len // 2):
    window = area[i-window_len//2:i+window_len//2+1]
    local_max = np.max(window)
    if(area[i] > local_max * 0.5):
      result[i] = 1.0
  return result

def post_process(result, txt_path):
  result_len = result.shape[0]
  ans = np.zeros(shape=result.shape)
  window_len = 5
  copy_data = np.zeros(shape = result_len+2*(window_len//2))
  for i in range(result_len):
    copy_data[i+window_len//2] = result[i]
  for i in range(window_len//2):
    w_l = window_len//2 - i
    copy_data[i] = copy_data[i+w_l*2]
    copy_data[-i-1] = copy_data[(-i-1)-w_l*2]

  result_len = copy_data.shape[0]

  for i in range(window_len//2, result_len-window_len//2):
    window = copy_data[i-window_len//2:i+window_len//2+1]
    result[i-window_len//2] = np.median(window)

  with open(txt_path, 'w') as f:
    for cnt, item in enumerate(result):
      f.write(str(item)+"\n")


if __name__ == '__main__':
  root = './S5_solution'
  data_path = {}
  for sequence in tqdm(os.listdir(f"{root}")):
    for file in os.listdir(f"{root}/{sequence}"):
      if file.endswith(".png"):
        if sequence not in data_path:
            data_path[sequence] = []
        data_path[sequence].append(f"{root}/{sequence}/{file}")
    area, avg_area = get_avg(data_path[sequence], f"{root}/{sequence}/")
    # print(area, avg_area)
    result = filter(area, avg_area)
    post_process(result, f"{root}/{sequence}/conf.txt")




  # img = cv2.imread(data_path["S5"]["01"][0])[:, :, 0]