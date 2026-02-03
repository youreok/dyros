import cv2
import os

def extract_first_frame(video_folder, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 영상 폴더 내의 파일 목록 가져오기
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_name in video_files:
        video_path = os.path.join(video_folder, video_name)
        
        # filename to save image
        file_name = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_folder, f"{file_name}.jpg")

        # open video
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            ret, frame = cap.read() # read first frame
            if ret:
                # save image
                cv2.imwrite(output_path, frame)
                print(f"추출 완료: {video_name} -> {file_name}.jpg")
            else:
                print(f"프레임을 읽지 못했습니다: {video_name}")
        else:
            print(f"영상을 열 수 없습니다: {video_name}")
            
        cap.release()

if __name__ == "__main__":
    VIDEO_DIR = "RoboTwin_Task_video"
    SAVE_DIR = "data/images"
    
    extract_first_frame(VIDEO_DIR, SAVE_DIR)