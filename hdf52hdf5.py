import h5py
import numpy as np
from tqdm import tqdm
import os
import cv2
from mmengine import fileio
import io

# 처리할 데이터셋 이름 목록
data_names = ['libero_10', 'libero_goal', 'libero_object', 'libero_90', 'libero_spatial']

# 각 데이터셋 이름에 대해 반복 처리
for data_name in data_names:
    # 사용할 이미지 키 (원본 데이터셋 기준)
    obs_keys = ['agentview_rgb', 'eye_in_hand_rgb']

    # --- 사용자 환경에 맞게 경로 수정 ---
    # base_dir: 변환할 원본 LIBERO 데이터셋 폴더 경로 (256x256 해상도 권장)
    # 예시: '/path/to/original/libero_datasets/256/'
    base_dir = f'/home/jihun/LIBERO/libero/datasets/{data_name}'

    # save_base_dir: 변환된 LBP 형식 HDF5 파일을 저장할 상위 폴더 경로
    # 예시: '/path/to/save/processed_libero_datasets/'
    # 각 data_name 별 하위 폴더는 스크립트가 생성함
    save_base_dir_root = '/home/jihun/LIBERO/libero/datasets_processed'
    save_base_dir = os.path.join(save_base_dir_root, data_name) # 데이터셋별 저장 폴더
    # --- 경로 수정 끝 ---

    print(f"\n--- Processing dataset: {data_name} ---")
    print(f"원본 경로: {base_dir}")
    print(f"저장 경로: {save_base_dir}")

    # 원본 경로 존재 확인
    if not os.path.isdir(base_dir):
        print(f"경고: 원본 경로를 찾을 수 없습니다. 건너<0xEB><0x9B><0x84>: {base_dir}")
        continue

    # 원본 경로 내 파일 목록 읽기
    try:
        hdf5_path_list = os.listdir(base_dir)
        if not hdf5_path_list:
            print(f"경고: 원본 경로에 파일이 없습니다: {base_dir}")
            continue
    except OSError as e:
        print(f"오류: 원본 경로를 읽을 수 없습니다 ({base_dir}): {e}. 건너<0xEB><0x9B><0x84>.")
        continue

    # 각 원본 HDF5 파일 처리
    for hdf5_path in tqdm(hdf5_path_list, desc=f"Processing files in {data_name}"):
        full_hdf5_path = os.path.join(base_dir, hdf5_path)

        # .hdf5 확장자 확인
        if not hdf5_path.endswith('.hdf5'):
            # print(f"알림: HDF5 파일이 아니므로 건너뜀: {hdf5_path}") # 너무 많은 로그 방지
            continue

        hdf5_file = None # 파일 객체 초기화 (finally에서 사용하기 위함)
        try:
            # 원본 파일 열기 시도
            hdf5_file = h5py.File(full_hdf5_path, 'r', swmr=False, libver='latest')

            # 'data' 그룹 존재 확인
            if "data" not in hdf5_file:
                print(f"경고: 'data' 그룹 없음 ({hdf5_path}). 파일 건너<0xEB><0x9B><0x84>.")
                continue # 다음 파일 처리

            demos = list(hdf5_file["data"].keys())

            # 유효한 데모 키 필터링 및 정렬 (demo_숫자 형식)
            valid_demos = [elem for elem in demos if elem.startswith('demo_') and elem[5:].isdigit()]
            if not valid_demos:
                print(f"경고: 유효한 데모 키 없음 ({hdf5_path}). 파일 건너<0xEB><0x9B><0x84>.")
                continue
            inds = np.argsort([int(elem[5:]) for elem in valid_demos])
            demos_sorted = [valid_demos[i] for i in inds]

            # 언어 지시 추출 (오류 방지 강화)
            lang_instruction = hdf5_path.replace("_demo.hdf5", "").replace("_", " ")
            lang_instruction = ''.join([char for char in lang_instruction if not (char.isupper() or char.isdigit())])
            lang_instruction = lang_instruction.strip() # 양 끝 공백 제거

            # 각 데모 처리
            for ep in tqdm(demos_sorted, desc=f"  Demos in {hdf5_path}", leave=False):
                try:
                    demo_group = hdf5_file[f"data/{ep}"]

                    # 필요한 키 존재 확인
                    required_obs_keys = ['agentview_rgb', 'eye_in_hand_rgb']
                    if 'obs' not in demo_group or not all(k in demo_group['obs'] for k in required_obs_keys):
                        print(f"경고: 필수 obs 키 부족 ({hdf5_path} / {ep}). 데모 건너<0xEB><0x9B><0x84>.")
                        continue
                    if 'actions' not in demo_group:
                        print(f"경고: 'actions' 키 없음 ({hdf5_path} / {ep}). 데모 건너<0xEB><0x9B><0x84>.")
                        continue
                    if 'robot_states' not in demo_group:
                        print(f"경고: 'robot_states' 키 없음 ({hdf5_path} / {ep}). 데모 건너<0xEB><0x9B><0x84>.")
                        continue

                    # 데이터 로드 (이미지는 uint8, 나머지는 float32)
                    third_image_data = demo_group["obs/agentview_rgb"][()].astype(np.uint8)
                    wrist_image_data = demo_group["obs/eye_in_hand_rgb"][()].astype(np.uint8)
                    action_data = demo_group["actions"][()].astype('float32')
                    state_data = demo_group["robot_states"][()].astype('float32')

                    # 데이터 유효성 확인 (빈 데이터 등)
                    if third_image_data.shape[0] == 0 or action_data.shape[0] == 0:
                         print(f"경고: 데이터 길이 0 ({hdf5_path} / {ep}). 데모 건너<0xEB><0x9B><0x84>.")
                         continue
                    if third_image_data.shape[0] != action_data.shape[0]:
                        print(f"경고: 이미지와 액션 길이 불일치 ({hdf5_path} / {ep}). 데모 건너<0xEB><0x9B><0x84>.")
                        continue


                    # 이미지 압축
                    compressed_3rd_img = []
                    compressed_wrist_img = []
                    valid_frames = True
                    for i in range(third_image_data.shape[0]):
                        # RGB -> BGR 변환 및 상하/좌우 반전 (LBP 스타일)
                        img_3rd_rgb = third_image_data[i]
                        img_3rd_bgr = cv2.cvtColor(img_3rd_rgb, cv2.COLOR_RGB2BGR)
                        img_3rd_flipped = img_3rd_bgr[::-1, ::-1] # Flip

                        # JPEG 압축 (품질 95)
                        result, encimg_3rd = cv2.imencode('.jpg', img_3rd_flipped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if not result:
                            print(f"경고: third_image 인코딩 실패 ({hdf5_path} / {ep} / frame {i}). 데모 건너<0xEB><0x9B><0x84>.")
                            valid_frames = False
                            break
                        compressed_3rd_img.append(encimg_3rd)

                        # 손목 카메라도 동일하게 처리 (RGB -> BGR)
                        img_wrist_rgb = wrist_image_data[i]
                        img_wrist_bgr = cv2.cvtColor(img_wrist_rgb, cv2.COLOR_RGB2BGR)
                        result, encimg_wrist = cv2.imencode('.jpg', img_wrist_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if not result:
                            print(f"경고: wrist_image 인코딩 실패 ({hdf5_path} / {ep} / frame {i}). 데모 건너<0xEB><0x9B><0x84>.")
                            valid_frames = False
                            break
                        compressed_wrist_img.append(encimg_wrist)

                    # 프레임 처리 중 오류 발생 시 해당 데모 건너뛰기
                    if not valid_frames:
                        continue

                    # 저장 경로 생성 (태스크 이름 폴더 / 데모 번호 파일)
                    task_name = hdf5_path.split('_demo.hdf5')[0]
                    save_dir = os.path.join(save_base_dir, task_name)
                    os.makedirs(save_dir, exist_ok=True)
                    file_save_path = os.path.join(save_dir, f"{ep}.hdf5")

                    # 메모리 내 HDF5 파일 생성 및 데이터 쓰기
                    f_mem = io.BytesIO()
                    with h5py.File(f_mem, 'w') as h_out:
                        g = h_out.create_group('observation')

                        # 압축된 이미지 저장 (가변 길이 데이터셋 사용)
                        dset_third = g.create_dataset('third_image', (len(compressed_3rd_img),), dtype=h5py.vlen_dtype(np.uint8))
                        for i, img_bytes in enumerate(compressed_3rd_img):
                            dset_third[i] = np.frombuffer(img_bytes, dtype=np.uint8) # Bytes -> numpy array

                        dset_wrist = g.create_dataset('wrist_image', (len(compressed_wrist_img),), dtype=h5py.vlen_dtype(np.uint8))
                        for i, img_bytes in enumerate(compressed_wrist_img):
                            dset_wrist[i] = np.frombuffer(img_bytes, dtype=np.uint8) # Bytes -> numpy array

                        # 액션, 상태(proprio), 언어 저장
                        h_out['action'] = action_data
                        h_out['proprio'] = state_data # 키 이름 변경
                        h_out['language_instruction'] = lang_instruction.encode('utf-8') # 문자열은 바이트로 인코딩

                    # 실제 파일로 저장 (mmengine 사용)
                    fileio.put(f_mem.getvalue(), file_save_path)

                # 데모 처리 중 발생한 예외 처리
                except Exception as demo_e:
                    print(f"!!! 데모 처리 오류 ({hdf5_path} / {ep}): {demo_e}. 해당 데모 건너<0xEB><0x9B><0x84>.")
                    continue # 다음 데모로 이동

        # 파일 열기 오류 처리
        except OSError as e:
            print(f"!!! 파일 열기/처리 오류 ({hdf5_path}): {e}. 파일 건너<0xEB><0x9B><0x84>.")
            continue # 다음 파일로 이동
        # 기타 예외 처리
        except Exception as e:
            print(f"!!! 예상치 못한 오류 ({hdf5_path}): {e}. 파일 건너<0xEB><0x9B><0x84>.")
            continue # 다음 파일로 이동
        # finally 블록: try 블록 실행 후 항상 실행 (오류 발생 여부와 관계없이)
        finally:
            if hdf5_file: # 파일 객체가 성공적으로 생성되었다면
                try:
                    hdf5_file.close() # 파일 닫기 시도
                except Exception as close_e:
                    print(f"경고: 파일 닫기 실패 ({hdf5_path}): {close_e}")

print("\n모든 데이터셋 처리 완료.")