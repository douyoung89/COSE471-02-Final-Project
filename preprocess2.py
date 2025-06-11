import os
import shutil
from collections import defaultdict # 이 코드는 사실 consolidate_augmented_data_folders에서는 직접 사용되지 않습니다만, 임포트 자체는 문제 없습니다.

def consolidate_augmented_data_folders(root_dir):
    """
    재구성된 Kaggle augmented data 폴더들을 'nameX-Y'에서 'nameX' 형태로 통합합니다.
    즉, 'name2-3', 'name2-4' 폴더 내의 파일들을 'name2' 폴더 아래로 옮깁니다.

    Args:
        root_dir (str): 'augmented train'과 같은 상위 디렉토리 경로.
                        이 디렉토리 아래에 'Agreeableness', 'Conscientiousness' 등의
                        특성별 폴더가 있으며, 그 안에 'nameX-Y' 형태의 폴더가 있습니다.
    """
    print(f"Starting folder consolidation in: {root_dir}")

    # 각 인격 특성(Agreeableness, Conscientiousness 등)별로 처리
    trait_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if not trait_dirs:
        print(f"No trait subdirectories found in {root_dir}. Please ensure trait folders exist.")
        return

    for trait in trait_dirs:
        current_trait_dir = os.path.join(root_dir, trait)
        print(f"\nProcessing trait: {trait} in {current_trait_dir}")

        # 현재 트레이트 디렉토리 내의 모든 하위 폴더 (nameX-Y 또는 other)를 순회
        subfolders = [f for f in os.listdir(current_trait_dir) 
                      if os.path.isdir(os.path.join(current_trait_dir, f))]
        
        if not subfolders:
            print(f"No subfolders found in {current_trait_dir}. Skipping.")
            continue

        # 이동 후 삭제할 빈 폴더들을 추적하기 위한 리스트
        folders_to_delete = []

        for subfolder_name in subfolders:
            original_subfolder_path = os.path.join(current_trait_dir, subfolder_name)
            
            # --- 핵심 변경 부분: 상위 그룹 ID 추출 로직 ---
            # 예: 'IMG_20200215_172314-7' -> 'IMG_20200215_172314'
            # 예: 'other' -> 'other' (하이픈 없으면 그대로)
            
            consolidated_group_id = subfolder_name
            if '-' in subfolder_name:
                # 첫 번째 하이픈을 기준으로 분리하여 첫 부분만 취함
                # 단, 'IMG_20200215_172314-7' 처럼 날짜+시간+숫자 형태로 되어있고
                # 마지막 하이픈 뒤의 숫자가 증강 넘버인 경우를 고려해야 합니다.
                # rsplit('-', 1)을 사용하여 마지막 하이픈 기준으로 한 번만 분리합니다.
                parts = subfolder_name.rsplit('-', 1) 
                if len(parts) > 1 and parts[-1].isdigit(): # 마지막 부분이 숫자라면 증강 넘버로 간주
                    consolidated_group_id = parts[0] # 앞 부분만 취함
                # else: # 마지막 부분이 숫자가 아니면 전체 폴더 이름이 ID (예: 'other')
                #     consolidated_group_id = subfolder_name # 이미 기본값으로 설정됨
            
            # --- 변경 끝 ---

            # 통합될 상위 그룹 폴더 경로 생성
            consolidated_group_path = os.path.join(current_trait_dir, consolidated_group_id)
            os.makedirs(consolidated_group_path, exist_ok=True) # 없으면 생성

            print(f"  Moving files from '{subfolder_name}' to '{consolidated_group_id}'...")

            # 현재 하위 폴더 내의 모든 파일들을 통합 폴더로 이동
            for filename in os.listdir(original_subfolder_path):
                file_path = os.path.join(original_subfolder_path, filename)
                if os.path.isfile(file_path): # 파일만 이동 (하위 폴더는 이동하지 않음)
                    # 여기가 오류가 발생했던 부분입니다. os.path.join으로 수정했습니다.
                    shutil.move(file_path, os.path.join(consolidated_group_path, filename))
            
            # 파일 이동이 완료된 후, 원래의 하위 폴더(nameX-Y)는 비어있게 되므로 삭제 목록에 추가
            folders_to_delete.append(original_subfolder_path)
            
        # 모든 파일 이동 후 빈 폴더 삭제
        for folder in folders_to_delete:
            try:
                if not os.listdir(folder): # 폴더가 비어있는지 다시 한번 확인
                    os.rmdir(folder)
                    print(f"  Removed empty folder: {folder}")
                else:
                    print(f"  Folder {folder} is not empty, skipping deletion.")
            except OSError as e:
                print(f"  Error deleting folder {folder}: {e}")

        print(f"Finished consolidating trait: {trait} folders.")

# 사용 예시:
source_augmented_data_root = "/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/archive-2/augmented test"

consolidate_augmented_data_folders(source_augmented_data_root)

print("\nFolder consolidation complete. Your folder structure should now be organized by 'nameX' groups.")