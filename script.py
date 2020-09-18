import torch
import sys
import json
from torchvision import transforms
import glob, os
from PIL import Image
import re
print('load model...')
model = torch.load('resner10111', map_location=torch.device('cpu'))  # load model

transform_val = transforms.Compose([  # transform needed size
    transforms.Resize([150, 150]),
    transforms.CenterCrop(150),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

DIR_NAME = str(sys.argv[1])  # get target file path


roots = glob.glob(DIR_NAME+'*.jpg')  # create massive with pathes

answer = {}
for name in roots:

    tens = transform_val(Image.open(name)).reshape([1, 3, 150, 150])  # transform photo
    name = name.split('\\')[-1]
    outputs = model(tens)  # model classifier
    _, label = torch.max(outputs.data, 1)  # get label
    answer[name] = 'male' if label.item() == 0 else 'female'  # 1 - female, 0- male

with open("process_results.json", "w") as fp:  # save json file
    json.dump(answer, fp)
    print('result was saved in {}process_results.json'.format(DIR_NAME))
