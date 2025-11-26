import torch
import clip
from PIL import Image
from torchvision import transforms
 
class ClipEmbeding:
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    def __init__(self):
        self.model, self.processor = clip.load("/home/xx/Projects/ID2/clip-model/ViT-B-16.pt", device=self.device)
        self.tokenizer = clip.tokenize
    '''
    def probs(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(["a diagram", "a horse", "a cat"]).to(self.device)
 
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
        print("Label probs:", probs)
 
    '''
    def embeding(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(process_image)
        return image_features
    
clip_embeding = ClipEmbeding()

import numpy as np
img_clip=[]
i=0
with open('/home/xx/Projects/DeepHash-pytorch-master/data/mirflickr/train.txt', 'r') as file:  
    while True:  
        line = file.readline()  
        i=i+1
        #print(i)
        if not line:  # 如果读取到文件末尾，则line为空字符串  
            break  
        # 处理每行数据
        a=line[:-77]  
        image_path="/hdd/public/datasets/flickr/img/mirflickr/"+a
        pil_image = Image.open(image_path)
        img_fea=clip_embeding.embeding(pil_image)
        img_clip.append(img_fea.squeeze(0).detach().cpu().numpy())
np.save('flickr_train_img.npy', img_clip) 