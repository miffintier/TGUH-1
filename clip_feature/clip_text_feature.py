import torch
import clip
from PIL import Image
from torchvision import transforms
import numpy as np
class ClipEmbeding:
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    def __init__(self):
        self.model, self.processor = clip.load("../clip-model/ViT-B-16.pt", device=self.device)
        self.tokenizer = clip.tokenize
    
    def embeding(self, text: str):
        text = self.tokenizer([text]).to(self.device)
 
        text_features = self.model.encode_text(text)
        #print(text_features)
        return text_features
clip_embeding = ClipEmbeding()
ofa_feature=[]
with open('../text_data/flickr/captions.txt', 'r', encoding='utf-8') as file:  
    while True:  
        # 读取一行  
        line = file.readline()  
        # 如果读取到文件末尾，则跳出循环  
        if not line:  
            break  
        # 处理读取到的行  
        text=line.strip()
        text_fea=clip_embeding.embeding(text)
        ofa_feature.append(text_fea.squeeze(0).detach().cpu().numpy())

#print(ofa_feature)
np.save('flickr_train_text.npy', ofa_feature) 

