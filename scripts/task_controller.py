import numpy as np
import json
import os

class TaskVectorTransformer:
    def __init__(self, object_data_dir="objects", results_dir="results"):
        self.object_data_dir = object_data_dir
        self.results_dir = results_dir

    def load_model_data(self, object_name):
        """objects/{name}/model_data1.json 로드"""
        path = os.path.join(self.object_data_dir, object_name, "model_data1.json")
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_adjoint_matrix(self, T):
        """Adjoint Transformation Matrix [Ad_T] 계산"""
        R = T[:3, :3]
        p = T[:3, 3]
        p_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[:3, 3:] = p_skew @ R
        adj[3:, 3:] = R
        return adj

    def compute_step_vector(self, step, T_world_hand):
        """각 Step의 로컬 벡터를 월드 좌표계로 변환"""
        actor_name = step.get("actor")
        if not actor_name or "vectorization" not in step:
            return None, None

        v_local = step["vectorization"]["V"]
        frame_mode = step["vectorization"]["frame_mode"]
        actor_pt = step.get("actor_point", {"id": 0, "type": "contact_point"})
        pt_id = actor_pt["id"]

        # 1. 모델 데이터 로드
        model_data = self.load_model_data(actor_name)
        if model_data is None:
            # 모델 데이터가 없는 경우 (bolt, screw 등)는 Hand Pose 기준 처리
            T_ref = T_world_hand
        else:
            # 2. Matrix 선택
            # contact_matrix가 리스트라고 가정: contact_matrix[pt_id]
            T_h_c = np.array(model_data["contact_matrix"][pt_id]) if "contact_matrix" in model_data else np.eye(4)
            
            if frame_mode == "CONTACT":
                T_ref = T_world_hand @ T_h_c
            elif frame_mode == "FUNCTIONAL":
                # functional_matrix도 리스트: functional_matrix[pt_id]
                T_c_f = np.array(model_data["functional_matrix"][pt_id]) if "functional_matrix" in model_data else np.eye(4)
                T_ref = T_world_hand @ T_h_c @ T_c_f
            else: # WORLD mode
                T_ref = np.eye(4)
                # World 모드에서 위치 정보가 필요하다면 T_ref의 translation만 유지하거나 규약에 따름
                T_ref[:3, 3] = (T_world_hand @ T_h_c)[:3, 3] 

        # 3. Adjoint 변환 수행
        adj = self.get_adjoint_matrix(T_ref)
        v_world = adj @ np.array(v_local)
        
        return v_world, T_ref

    def run_analysis(self):
        print("\n=== RoboTwin Vector Transformation Analysis ===")
        task_name = input("Enter Task Name (e.g., Screwing A Screw): ").strip()
        json_path = os.path.join(self.results_dir, f"{task_name}.json")

        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found.")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)

        # 발표용 가상 Hand Pose 설정 (필요시 수정)
        T_virtual_hand = np.eye(4)
        T_virtual_hand[:3, 3] = [0.5, 0.2, 0.8] # 가상 위치

        print(f"\n[Task: {task_name}] Processing sequence...")
        print("-" * 50)

        for step in plan.get("sequence", []):
            v_world, T_ref = self.compute_step_vector(step, T_virtual_hand)
            
            print(f"Step {step['step']}: {step['intent']}")
            print(f" - Subtask: {step['subtask']}")
            if v_world is not None:
                print(f" - Frame: {step['vectorization']['frame_mode']} (ID: {step.get('actor_point', {}).get('id', 0)})")
                print(f" - World Velocity (v): {np.round(v_world[:3], 4)}")
                print(f" - World Angular  (w): {np.round(v_world[3:], 4)}")
            else:
                print(" - No vectorization data for this step.")
            print("-" * 50)

if __name__ == "__main__":
    transformer = TaskVectorTransformer()
    transformer.run_analysis()