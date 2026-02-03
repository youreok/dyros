import pandas as pd
import json
import os
import base64
from openai import OpenAI

# 1. 초기 설정
client = OpenAI(api_key="YOUR_API_KEY_HERE") 
CSV_PATH = "data/RoboTwin_Task.csv"
IMAGE_DIR = "data/images"
PROMPT_PATH = "prompts/system_prompt.txt"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_manual_test():
    # 2. 엑셀 데이터 로드
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} is wrong dir")
        return
    df = pd.read_csv(CSV_PATH)
    
    # 3. 시스템 프롬프트 로드
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("\n=== Load Dataset ===")
    search_name = input("Task Name: ").strip()

    # 4. 엑셀에서 해당 태스크 정보 찾기
    task_info = df[df['Tasks'].str.lower() == search_name.lower()]

    if task_info.empty:
        print(f"'{search_name}'is wrong task")
        return

    # 정보 추출
    task_name = task_info.iloc[0]['Tasks']
    description = task_info.iloc[0]['Description']
    image_path = os.path.join(IMAGE_DIR, f"{task_name}.jpg") # 확장자는 상황에 맞게 (.png 등) 수정

    print(f"\n== Task ==")
    print(f"- Task: {task_name}")
    print(f"- Description: {description}")
    print(f"- Image Path: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: There is no {image_path}")
        return

    # 5. GPT-4o 호출
    print(f"\n...Analyzing...")
    try:
        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Task: {task_name}\nDescription: {description}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )

        result_json = json.loads(response.choices[0].message.content)
        print("\n=== Sub-task Sequence ===")
        print(json.dumps(result_json, indent=2, ensure_ascii=False))

        # 파일 저장
        save_path = f"{OUTPUT_DIR}/{task_name}_result.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_manual_test()