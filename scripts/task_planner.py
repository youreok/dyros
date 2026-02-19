import pandas as pd
import json
import os
import base64

from validator import validate_plan, build_point_id_index, issues_to_text
from make_report import save_reports

from openai import OpenAI

client = OpenAI(api_key="")  # Set your API key here or via environment variable

# -------- robust paths --------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH = os.path.join(ROOT_DIR, "data", "RoboTwin_Task.csv")
OBJECT_DATA_DIR = os.path.join(ROOT_DIR, "objects")
PROMPT_PATH = os.path.join(ROOT_DIR, "prompts", "system_prompt.txt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_name(name):
    return name.strip().strip('"').strip("'")


def load_object_points(object_names_str):
    combined_info = ""
    names = [clean_name(n) for n in object_names_str.split(",")]
    for name in names:
        path = os.path.join(OBJECT_DATA_DIR, name, "points_info.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_info += f"\n[Object: {name}]\n{json.dumps(data, indent=2, ensure_ascii=False)}\n"
    return combined_info


def run_task_planner():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} is missing")
        return None, None

    # CSV 로딩
    df = pd.read_csv(CSV_PATH, quotechar='"', skipinitialspace=True)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("\n=== RoboTwin Strategic Planner (Multi-Image Mode) ===")
    search_name = input("Task Name (from CSV): ").strip()

    task_info = df[df["Tasks"].astype(str).str.lower() == search_name.lower()]
    if task_info.empty:
        print(f"'{search_name}' is not in the dataset.")
        return None, None

    task_name = str(task_info.iloc[0]["Tasks"])
    description = str(task_info.iloc[0]["Description"])
    objects_str = str(task_info.iloc[0]["Objects"])
    object_list = [clean_name(o) for o in objects_str.split(",")]

    # 1) user content: text + all object images
    user_content = [
        {
            "type": "text",
            "text": (
                f"Task: {task_name}\n"
                f"Description: {description}\n\n"
                f"Object Data (Metadata):\n{load_object_points(objects_str)}"
            ),
        }
    ]

    found_images = []
    for obj in object_list:
        img_path = os.path.join(OBJECT_DATA_DIR, obj, "image.jpg")
        if os.path.exists(img_path):
            found_images.append(img_path)
            base64_img = encode_image(img_path)
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            )

    if not found_images:
        print(f"Error: No image.jpg found for objects {object_list}")
        return None, None

    print(f"- Task: {task_name}")
    print(f"- Images loaded: {found_images}")
    print("\n...Analyzing Visual Context (Multi-View) & Screw Vectors...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        result_json = json.loads(response.choices[0].message.content)

        # --- Load points_info for validation
        points_info_by_object = {}
        for obj in object_list:
            ppath = os.path.join(OBJECT_DATA_DIR, obj, "points_info.json")
            if os.path.exists(ppath):
                with open(ppath, "r", encoding="utf-8") as f:
                    points_info_by_object[obj] = json.load(f)

        point_index = build_point_id_index(points_info_by_object)

        # --- Validate + sanitize
        val = validate_plan(result_json, point_index, auto_fix=True)

        # --- Save raw + validated
        raw_path = os.path.join(OUTPUT_DIR, f"{task_name}__raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

        save_path = os.path.join(OUTPUT_DIR, f"{task_name}.json")  # validated
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(val.sanitized, f, indent=2, ensure_ascii=False)

        # --- Print issues
        if val.issues:
            print("\n[Validator Issues]")
            print(issues_to_text(val.issues))
        else:
            print("\n[Validator] No issues.")

        # --- Save CSV/TXT reports (no PNG)
        report_paths = save_reports(task_name, result_json, val, output_dir=OUTPUT_DIR)
        print("\n[Reports saved]")
        for k, v in report_paths.items():
            print(f"- {k}: {v}")

        return val.sanitized, objects_str

    except Exception as e:
        print(f"Error during LLM Inference: {e}")
        return None, None


if __name__ == "__main__":
    result, target_objects = run_task_planner()
