import os
import shutil

def reorganize_augmented_data(root_dir):
    """
    Kaggle augmented data의 파일들을 원본 이미지 그룹별로 하위 폴더로 재구성합니다.
    예: 'name1-1.png', 'name1-1-1.png' -> 'name1' 폴더 아래로 이동

    Args:
        root_dir (str): 'augmented train'과 같은 상위 디렉토리 경로.
                        이 디렉토리 아래에 'Agreeableness', 'Conscientiousness' 등의
                        특성별 폴더가 있어야 합니다.
    """
    print(f"Starting data reorganization in: {root_dir}")

    # 각 인격 특성(Agreeableness, Conscientiousness 등)별로 처리
    trait_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if not trait_dirs:
        print(f"No subdirectories found in {root_dir}. Please ensure trait folders exist.")
        return

    for trait in trait_dirs:
        current_trait_dir = os.path.join(root_dir, trait)
        print(f"\nProcessing trait: {trait} in {current_trait_dir}")

        # 모든 파일 경로를 먼저 수집
        all_files_in_trait = []
        for filename in os.listdir(current_trait_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files_in_trait.append(os.path.join(current_trait_dir, filename))
        
        if not all_files_in_trait:
            print(f"No image files found in {current_trait_dir}. Skipping.")
            continue

        # 원본 이미지 그룹 ID별로 파일들을 새 폴더로 이동
        for file_path in all_files_in_trait:
            filename = os.path.basename(file_path)
            base_name_without_ext = os.path.splitext(filename)[0]

            # 'name1-1.png', 'name1-1-1.png', 'name1-1-5.png' -> original_group_id = 'name1'
            # 'name2-1.png', 'name2-1-1.png', 'name2-1-5.png' -> original_group_id = 'name2'
            
            # 패턴: '작가이름-글번호[-증강번호].확장자'
            # 목표: '작가이름-글번호' 부분만 추출하여 그룹 ID로 사용
            
            parts = base_name_without_ext.split('-')
            
            original_group_id = base_name_without_ext # 기본값 설정

            # 만약 파일명이 '작가이름-글번호-증강번호' 형태라면, 증강번호 부분만 제거
            # 예: 'anuraggupta19eng-1-1' 에서 'anuraggupta19eng-1'을 추출
            if len(parts) >= 3 and parts[-1].isdigit(): # 최소한 'name-id-aug' 형태
                # 'anuraggupta19eng-1-1' -> ['anuraggupta19eng', '1', '1']
                # Join all but the last part
                original_group_id = '-'.join(parts[:-1])
            elif len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit(): # 예: 'name-1-1' 같이 마지막 두 파트가 숫자인 경우
                # 다시 한번 'anuraggupta19eng-1-1' 에서 'anuraggupta19eng-1'을 추출하기 위해
                # 마지막 파트가 숫자라면 증강 번호로 간주하여 제거
                # 이 로직은 `anuraggupta19eng-1.jpg`와 `anuraggupta19eng-1-1.jpg`를 모두 `anuraggupta19eng-1`로 묶습니다.
                if '-' in base_name_without_ext:
                    split_by_dash = base_name_without_ext.rsplit('-', 1) # 마지막 하이픈 기준으로 한 번만 분리
                    if split_by_dash[-1].isdigit(): # 마지막 부분이 숫자라면 증강 넘버로 간주
                        original_group_id = split_by_dash[0] # 앞 부분만 취함
                    else: # 마지막 부분이 숫자가 아니면 전체가 원본 ID (예: name-ID.jpg)
                        original_group_id = base_name_without_ext
                else: # 하이픈이 없으면 전체가 원본 ID (예: name.jpg)
                    original_group_id = base_name_without_ext
            
            # 여기서 original_group_id가 최종적으로 'name1', 'name2' 등이 됩니다.

            # 새 그룹 폴더 경로 생성
            new_group_dir = os.path.join(current_trait_dir, original_group_id)
            os.makedirs(new_group_dir, exist_ok=True) # 폴더 없으면 생성

            # 파일 이동
            shutil.move(file_path, os.path.join(new_group_dir, filename))
            # print(f"Moved {filename} to {new_group_dir}")

        print(f"Finished reorganizing trait: {trait}. Files are now grouped into folders.")

# 사용 예시:
# 이 경로는 'augmented train' (또는 모든 증강된 데이터를 담고 있는 상위 폴더)를 지정합니다.
source_augmented_data_root = "/Users/douyoung/Library/CloudStorage/GoogleDrive-douyoung@gmail.com/내 드라이브/1. 고려대학교/4학년/1학기/데이터과학/archive-2/augmented test"

reorganize_augmented_data(source_augmented_data_root)

print("\nData reorganization complete. Your folder structure should now be:")
print(f"{source_augmented_data_root}/Agreeableness/name1/... and so on.")