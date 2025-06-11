import pandas as pd 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json 
######## 추가한 부분 ########
import cv2
import numpy as np
###########################
import os
from PIL import Image
import torchvision.transforms as transforms
from cnn import PersonalClassifier, PersonalityDataset, train_valid_split, test


if __name__ == '__main__':
    model = PersonalClassifier()
    # model = PersonalClassifier(in_channels=4, num_classes=5) # adding edge : channel 3 to 4 
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model.load_state_dict(torch.load("/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/datascience/final proj/models/personality_cnn.pt", weights_only=True, map_location=torch.device(device)))
    # Evaluating phase 
    test_csv = "/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/archive-2/augmented test"
    testset = PersonalityDataset(test_csv,)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    test(model=model, optimizer=optimizer, loss_func=loss_func, testloader=testloader, dataset=testset, device=device)
