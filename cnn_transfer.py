import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models # torchvision.models 추가
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random 

# PersonalityDataset 클래스는 동일하게 유지
class PersonalityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.label_map = {
            'Agreeableness': 0,
            'Conscientiousness': 1,
            'Extraversion': 2,
            'Neuroticism': 3,
            'Openness': 4
        }

        # --- 이 부분이 핵심적인 변경 사항입니다 ---
        self.data_by_group = {} # 그룹 ID별 데이터 저장
        self.data = [] # 전체 데이터 (원본 그룹 ID 포함)
        # --- 여기까지 ---

        # 각 인격 특성 폴더 (Agreeableness, Conscientiousness 등)를 순회
        # 폴더 구조 : 
        #         <root_dir>/
        # ├── Agreeableness/         # trait 폴더 (label_map의 키)
        # │   ├── nameA/             # original_id_group 폴더 (group_id)
        # │   │   ├── img1.png
        # │   │   ├── img1-1.png
        # │   │   └── img1-2.png
        # │   ├── nameB/
        # │   │   ├── img2.jpg
        # │   │   └── img2-3.jpg
        # │   └── …  
        # ├── Conscientiousness/
        # │   ├── nameC/
        # │   │   ├── C.png
        # │   │   └── C-2.png
        # │   └── …  
        # ├── Extraversion/
        # │   └── …  
        # ├── Neuroticism/
        # │   └── …  
        # └── Openness/
        #     └── …  
        for trait, label in self.label_map.items():
            trait_path = os.path.join(self.root_dir, trait)
            
            if not os.path.isdir(trait_path):
                print(f"Warning: Trait directory not found: {trait_path}. Skipping.")
                continue

            # trait_path 내부의 각 'nameX' (Original_ID_Folder) 폴더를 순회
            for original_id_folder_name in os.listdir(trait_path):
                original_id_folder_path = os.path.join(trait_path, original_id_folder_name)
                
                if os.path.isdir(original_id_folder_path):
                    # 원본 ID (폴더 이름)를 그룹 ID로 사용
                    group_id = original_id_folder_name 
                    if group_id not in self.data_by_group:
                        self.data_by_group[group_id] = []
                    

                    # 'nameX' 폴더 내부의 이미지 파일들을 순회
                    for fname in os.listdir(original_id_folder_path):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            fpath = os.path.join(original_id_folder_path, fname)
                            self.data_by_group[group_id].append((fpath, label, group_id)) # 그룹 ID도 함께 저장
                            self.data.append((fpath, label, group_id)) # self.data에도 그룹 ID를 함께 저장
                            

        if not self.data:
            print(f"No image data loaded from {root_dir}. Please check the directory structure and file types.")
        else:
            print(f"Successfully loaded {len(self.data)} images from {root_dir}.")
            # --- 이 부분도 추가하면 좋습니다 ---
            print(f"Loaded from {len(self.data_by_group)} unique original ID folders.")
            # --- 여기까지 ---

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, _ = self.data[idx] # 그룹 ID는 __getitem__에서는 사용하지 않음
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()

# train_valid_split 함수는 동일하게 유지
def train_valid_split(dataset, valid_ratio=0.2):
    all_group_ids = list(dataset.data_by_group.keys())
    random.shuffle(all_group_ids) # 그룹 ID를 무작위로 섞음

    num_valid_groups = int(len(all_group_ids) * valid_ratio)
    valid_group_ids = all_group_ids[:num_valid_groups]
    train_group_ids = all_group_ids[num_valid_groups:]

    train_data_indices = []
    valid_data_indices = []

    for i, (img_path, label, group_id) in enumerate(dataset.data):
        if group_id in train_group_ids:
            train_data_indices.append(i)
        elif group_id in valid_group_ids:
            valid_data_indices.append(i)
    
    # Subset 클래스를 사용하여 데이터셋 분할
    train_subset = torch.utils.data.Subset(dataset, train_data_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_data_indices)

    print(f"Total groups: {len(all_group_ids)}")
    print(f"Train groups: {len(train_group_ids)}, Train samples: {len(train_subset)}")
    print(f"Valid groups: {len(valid_group_ids)}, Valid samples: {len(valid_subset)}")

    return train_subset, valid_subset

# train_and_valid 함수는 동일하게 유지
def train_and_valid(model, optimizer, loss_func, trainloader, validloader, epochs=10, device="cpu", save_path = "./personality_cnn.pt", early_stopping_limit=3):
    model.to(device)

    tr_loss_lst = []
    vl_loss_lst = []
    best_valid = float('inf')
    patience_counter = 0 
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for (feature, label) in tqdm(trainloader, desc="Training"):
            feature, label = feature.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(feature)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(trainloader)
        tr_loss_lst.append(train_loss)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad() :
            for (feature, label) in tqdm(validloader, desc="Validating"):
                feature, label = feature.to(device), label.to(device)
                output = model(feature)

                _, predicted = torch.max(output.data, 1)
                loss = loss_func(output, label)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                val_loss += loss.item()

        val_loss = val_loss / len(validloader)
        vl_loss_lst.append(val_loss)
        accuracy = correct / total


        print(f"""Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.4f} Valid Loss: {val_loss:.4f}""")
        print(f'Validation Accuracy: {accuracy:.4f}')

        if best_valid > val_loss :
            best_valid = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best valid loss: {best_valid:.4f} | Current valid loss: {val_loss:.4f}") # 형식 통일
            print(f"Model saved to {save_path}")
            print()
        else :
            patience_counter += 1
            print("Skip saving model")
            print()
            if patience_counter >= early_stopping_limit:
                print("Early stopping triggered")
                print()
                break

    return tr_loss_lst, vl_loss_lst

# loss_graph 함수는 동일하게 유지
def loss_graph(train, valid, save_path="loss_curve_cnn.png"):
    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label='Train Loss', marker='o')
    plt.plot(epochs, valid, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Loss graph saved to {save_path}")

# --- 전이 학습을 위한 모델 정의 (PersonalClassifier 대신) ---
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TransferLearningModel, self).__init__()
        # 사전 학습된 ResNet-18 모델 로드
        self.resnet = models.resnet50(pretrained=True)

        # ResNet의 모든 파라미터를 고정 (feature extractor)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = True

        # 마지막 fully connected layer를 새로운 클래스 수에 맞게 수정
        # ResNet-18의 마지막 FC 레이어의 입력 특징(in_features)을 확인
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3), # 여기에 드롭아웃 추가
            nn.Linear(num_ftrs, num_ftrs), 
            nn.GELU(), 
            nn.Dropout(0.3), # 여기에 드롭아웃 추가
            nn.Linear(num_ftrs, num_classes) 
        )

    def forward(self, x):
        return self.resnet(x)

## evaluating 함수는 동일하게 유지
def test(model, optimizer, loss_func, testloader, dataset, device="cpu", save_path="confusion_matrix_cnn.png", score_path="scores_cnn.json"):
    model.to(device)
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad() :
        for (feature, label) in tqdm(testloader, desc="Evaluating"):
            feature, label = feature.to(device), label.to(device)

            output = model(feature)
            _, predicted = torch.max(output.data, 1)
            loss = loss_func(output, label)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            val_loss += loss.item()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        accuracy = correct / total
        loss = val_loss / len(testloader)
        report_dict = classification_report(y_true, y_pred, target_names=list(dataset.label_map.keys()), output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        labels = list(dataset.label_map.keys())

        # show in kernerl
        print(f"""Loss: {loss:.4f}""")
        print(f'Test Accuracy: {accuracy:.4f}')
        print(classification_report(y_true, y_pred, target_names=list(dataset.label_map.keys())))

        # save confusion matrix as png
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

        # save scores as json file
        score_data = {
        "loss": loss,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
        }
        with open(score_path, "w") as f:
            json.dump(score_data, f, indent=2)
        print(f"Scores saved to {score_path}")

    return loss, accuracy

if __name__ == "__main__":
    # 데이터셋 경로 설정 (이 부분은 사용자 환경에 맞게 수정해주세요)
    # 예시: path = "C:/Users/username/Desktop/augmented_train"
    path = "/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/archive-2/augmented train"
    dataset = PersonalityDataset(path)
    train_dataset, test_dataset = train_valid_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # PersonalClassifier 대신 TransferLearningModel 사용
    model = TransferLearningModel(num_classes=len(dataset.label_map)) # 클래스 수는 label_map의 크기로
    loss_func = nn.CrossEntropyLoss()
    # 전이 학습에서는 새로 추가된 FC 레이어의 파라미터만 학습시키는 것이 일반적
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # ResNet의 FC 레이어만 학습

    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")


    train_loss, valid_loss = train_and_valid(model=model,
                                            optimizer=optimizer,
                                            loss_func=loss_func,
                                            trainloader=train_loader,
                                            validloader=valid_loader,
                                            epochs=epochs,
                                            device=device,
                                            save_path = "./personality_transfer_learning.pt", # 저장 경로 변경
                                            early_stopping_limit=5 
                                            )

    loss_graph(train_loss, valid_loss, save_path="loss_curve_transfer_learning.png") # 그래프 저장 경로 변경


    # 저장된 가중치 로드
    model.load_state_dict(torch.load("personality_transfer_learning.pt"))
    # Evaluating phase
    # test_csv 경로 설정 (이 부분도 사용자 환경에 맞게 수정해주세요)
    # 예시: test_csv = "C:/Users/username/Desktop/augmented_test"
    test_csv = "/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/archive-2/augmented test"
    testset = PersonalityDataset(test_csv)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    test(model=model, optimizer=optimizer, loss_func=loss_func, testloader=testloader, dataset=testset,device=device,
         save_path="confusion_matrix_transfer_learning.png", score_path="scores_transfer_learning.json") # 저장 경로 변경