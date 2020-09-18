import torch
import sys
import json
from torchvision import transforms
print('load model...')
model  = torch.load('resner10111',map_location=torch.device('cpu'))
transform_val = transforms.Compose([
         transforms.Resize([150,150]),
         transforms.CenterCrop(150),

         transforms.ToTensor(),
         transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])

])
DIR_NAME = str(sys.argv[1])
import glob, os
from PIL import Image
os.chdir(DIR_NAME )
roots = glob.glob('*.jpg')

answer = {}
for name in roots:
  tens = transform_val(Image.open(name)).reshape([1,3,150,150])
  outputs= model(tens)
  _,label =torch.max(outputs.data, 1)
  answer[name] = 'male' if label.item()==0 else 'female'


with open("process_results.json", "w") as fp:
    json.dump(answer,fp)
    print('result was saved in {}process_results.json'.format(DIR_NAME))

