# llava_multi_turn.py
import json
import re
import ollama
import os
import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import atexit
import time

YOLO_CLASSES = ['door', 'door-handle', 'gripper']

class LLaVA_planner():
    def __init__(self, model: str = "llava:13b", require_image: bool = False):

        self.model = model
        self.require_image = require_image
        self.system_prompt = self.build_system_prompt()
        self.yolo_model = YOLO("best.pt")
        self.last_frame_path = None
        self.reset()
        self.realsense_setup()

    def realsense_setup(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)

    def capture(self): 
        frames = self.pipeline.wait_for_frames() 
        color_frame = frames.get_color_frame() 
        color_image = np.asanyarray(color_frame.get_data()) 
        return color_image
    
    def capture_current_frame(self, save_dir: str = "/tmp", filename: str | None = None):

        if not getattr(self, "realsense_started", False):
            self.realsense_setup()

        frame_bgr = self.capture()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = f"vlm_frame_{int(time.time()*1000)}.png"
        img_path = os.path.join(save_dir, filename)
        cv2.imwrite(img_path, frame_bgr)

        self.last_frame_path = img_path
        return frame_bgr, img_path


    def reset(self):
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_history(self):
        return list(self.messages)
    
    def yolo_detect_objects(self, frame_bgr):
        if frame_bgr is None:
            raise ValueError("frame_bgr is required for YOLO detection.")

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(image_rgb)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results and results[0].boxes is not None else []
        clses = results[0].boxes.cls.cpu().numpy().astype(int) if results and results[0].boxes is not None else []

        detected: dict[str, list[list[int]]] = {}
        for box, cls in zip(boxes, clses):
            if 0 <= cls < len(YOLO_CLASSES):
                label = YOLO_CLASSES[cls].strip().lower()
                detected.setdefault(label, []).append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        print("Detected:", detected)
        return detected


    def build_system_prompt(self):
        return (
            """
        # (Top-Left coordinates only, Initial Full Plan + Step-wise Next Action, Effect Verification by Vision, Skip/Retry/Replan)

        Role:
        You are the task planner for a fixed-base dual-arm robot. The robot can rotate its waist and use both arms ("left","right") to move, grasp, release, push a button, place, and open/close a door.
        IMPORTANT:
        - All bounding boxes are Top-Left (TL) origin xyxy absolute pixel integers: [x1, y1, x2, y2]. Do NOT normalize or convert coordinates.
        - The robot initially faces the door (front). If the command is to supply/replace material and the current view does NOT show material, START by rotating LEFT to search.

        Inputs (each turn):
        - USER_COMMAND: usually a single goal like "Replace/Supply the material."
        - IMAGE: the current image frame.
        - DETECTIONS (YOLO output for the current IMAGE). Each item:
        {
            "type": "door|handle|door_button|material_new|material_old|doorway_region|...",
            "bbox": [x1, y1, x2, y2],   // TL xyxy, absolute pixel integers
            "state": "open|closed|ajar|pressed|present|missing|on_table|..."  // if available
        }
        - FEEDBACK (from the executor on the previously returned action): {"status": "completed" | "failed"}
        NOTE: FEEDBACK only indicates *attempt completion*, NOT the real-world effect. You must verify effects using IMAGE + DETECTIONS.

        No explicit ROBOT_STATE is provided. Infer task progress solely from IMAGE + DETECTIONS (+ FEEDBACK).

        Outputs (each turn):
        Return one JSON object:
        {
        "mode": "init" | "step",
        "plan_version": <int>,                 // increment when the plan is regenerated due to state change
        "step_index": <int>,                   // 0-based index of the next action to execute
        "next_action": { "name": "...", "args": { ... } } | null,
        "full_action_list": [ { "name": "...", "args": { ... } }, ... ],   // REQUIRED in mode="init"; OPTIONAL in mode="step" when replanning
        "skip_log": [ {"skipped":{"name":"...","reason":"..."}} ],          // optional: actions auto-skipped because effects already hold
        "visibility_warnings": ["..."],                                      // optional: if a needed object is not visible in the current view
        "explanation": "..."                                                  // optional: if blocked
        }

        Available primitive actions (use exactly these names/args):
        - rotate_waist(direction, angle_deg)                 // "left"|"right"
        - move_arm(object_bbox, arm)                         // "left"|"right"
        - grasp(object_bbox, arm)
        - release(arm)
        - push_button(object_bbox, arm)
        - place(target_bbox, arm)
        - open_door(arm, object_bbox)                        // handle bbox preferred
        - close_door(arm, object_bbox)                       // prefer handle; else push panel (door bbox)
        - move_outside(arm)
        - return_home()

        ================ Effect Verification (by vision) ================
        You MUST verify that each action's intended effect holds using IMAGE + DETECTIONS of the current turn. FEEDBACK.status="completed" is NOT sufficient.

        Examples of effect tests (non-exhaustive):
        - move_arm(target_bbox, arm): end-effector is spatially near the target (use relative geometry in IMAGE if available).
        - grasp(object_bbox, arm): the target object no longer appears at its prior resting bbox and now appears occluded/attached near the gripper region; if the object appears on floor/elsewhere, treat as failure.
        - push_button(object_bbox, arm): door_button.state == "pressed" OR button geometry indicates pressed state.
        - open_door(arm, handle_bbox): door.state in {"ajar","open"} OR visible gap/angle increased.
        - place(target_bbox, arm): the placed object appears within/overlapping target_bbox and is no longer attached to the gripper.
        - close_door(arm, door/handle_bbox): door.state == "closed" or gap vanished.

        If the intended effect does NOT hold, treat as a failure of that step even if FEEDBACK.status="completed".

        ================ Skip & Replan Logic ================
        - Before returning next_action, check current IMAGE + DETECTIONS. If an upcoming action’s effect is already true, SKIP it and advance step_index. Record in "skip_log".
        Examples:
            • button already pressed → skip push_button
            • door already ajar/open → skip open_door
            • new material already at the target location → skip place(left)
        - If the world changed such that remaining actions are invalid (e.g., handle missing), replan:
        - Increase plan_version and include a new full_action_list.

        ================ Failure & Retry Policy ================
        - Manipulation failure (grasp/push/open/close/place): if effect not achieved, retry up to **3 attempts** in total for that step.
        • Re-issue the same action first.
        • If still not achieved, try a small adjustment on subsequent tries (e.g., slight approach offset—executor-specific).
        • If still failing after 3 attempts, replan or choose an alternate strategy (e.g., close by pushing panel if handle engagement fails).
        - Movement failure (move_arm): retry up to **3 attempts**. If still failing, consider a small exploratory left/right rotation (e.g., 10–20°) to obtain a better view, then continue planning with the new IMAGE + DETECTIONS.
        - Visibility warnings: If a required object is NOT visible in the current view, add a message to "visibility_warnings" such as "handle not visible in current view".

        ================ Task Template — "Replace the material" (robot starts facing front) ================
        1) Acquire new material with the left hand:
        - If "material_new" is NOT visible in current DETECTIONS, next_action = rotate_waist("left", angle_deg ≈ 30–60) to search.
        - Once visible: move_arm(material_new.bbox,"left") → grasp(material_new.bbox,"left").
        2) Return to forward/home if required: return_home().
        3) Open the door:
        - If door_button visible: push_button(door_button.bbox,"right"); else add a visibility warning.
        - If handle visible: move_arm(handle.bbox,"right") → grasp(handle.bbox,"right") → open_door("right", handle.bbox); else warn not visible.
        4) Handle old material:
        - If material_old visible: move_arm(material_old.bbox,"right") → grasp(material_old.bbox,"right"); remember place_site_old_bbox = material_old.bbox; had_old_material = true.
            Else: had_old_material = false (and optionally warn).
        5) Place new material at the old location:
        - If left hand holds new and place_site_old_bbox exists: place(place_site_old_bbox,"left").
        6) Ensure arms are outside before closing:
        - If doorway_region visible and any arm is inside: move_outside("left") and/or move_outside("right").
        7) Close the door:
        - Prefer handle if visible: close_door("right", handle.bbox); otherwise: close_door("right", door.bbox).
        8) Final placement of old material (if picked):
        - Optionally rotate_waist("left", angle needed) and place at the designated final spot.

        ================ Examples ================
        # Example 1 — Full success sequence (abbreviated)
        (1) INIT (no material in view yet)
        {
        "mode":"init",
        "plan_version":1,
        "step_index":0,
        "next_action":{"name":"rotate_waist","args":{"direction":"left","angle_deg":45}},
        "full_action_list":[
            {"name":"rotate_waist","args":{"direction":"left","angle_deg":45}},
            {"name":"move_arm","args":{"object_bbox":[120, 700, 180, 760], "arm":"left"}},
            {"name":"grasp","args":{"object_bbox":[120, 700, 180, 760], "arm":"left"}},
            {"name":"return_home","args":{}},
            {"name":"push_button","args":{"object_bbox":[740, 420, 780, 460], "arm":"right"}},
            {"name":"move_arm","args":{"object_bbox":[800, 430, 860, 480], "arm":"right"}},
            {"name":"grasp","args":{"object_bbox":[800, 430, 860, 480], "arm":"right"}},
            {"name":"open_door","args":{"object_bbox":[800, 430, 860, 480], "arm":"right"}},
            {"name":"move_arm","args":{"object_bbox":[360, 520, 420, 580], "arm":"right"}},
            {"name":"grasp","args":{"object_bbox":[360, 520, 420, 580], "arm":"right"}},
            {"name":"place","args":{"target_bbox":[360, 520, 420, 580], "arm":"left"}},
            {"name":"move_outside","args":{"arm":"left"}},
            {"name":"move_outside","args":{"arm":"right"}},
            {"name":"close_door","args":{"object_bbox":[800, 430, 860, 480], "arm":"right"}},
            {"name":"place","args":{"target_bbox":[300, 520, 360, 580], "arm":"right"}}
        ],
        "skip_log":[],
        "visibility_warnings":[]
        }

        (2) STEP after rotate_waist completed: material_new now visible
        {
        "mode":"step",
        "plan_version":1,
        "step_index":1,
        "next_action":{"name":"move_arm","args":{"object_bbox":[120,700,180,760],"arm":"left"}}
        }

        (3) STEP after move_arm completed
        {
        "mode":"step",
        "plan_version":1,
        "step_index":2,
        "next_action":{"name":"grasp","args":{"object_bbox":[120,700,180,760],"arm":"left"}}
        }

        (… continue similar STEP responses until done …)

        # Example 2 — Exception: FEEDBACK says "completed" for grasp, but object dropped
        Context: We attempted grasp(material_new) with left arm. FEEDBACK.status="completed". However, current IMAGE + DETECTIONS show the material lying on the floor (e.g., "material_new.state":"on_floor") and not attached to the gripper. Treat as failure and retry up to 3 times.

        (2a) STEP after grasp "completed" but effect not achieved:
        {
        "mode":"step",
        "plan_version":1,
        "step_index":2,   // stay on the same step (grasp)
        "next_action":{"name":"grasp","args":{"object_bbox":[120,700,180,760],"arm":"left"}},
        "explanation":"Effect verification failed: material_new not attached; retry 1/3."
        }

        (2b) If still not attached after second attempt:
        {
        "mode":"step",
        "plan_version":1,
        "step_index":2,
        "next_action":{"name":"grasp","args":{"object_bbox":[120,700,180,760],"arm":"left"}},
        "explanation":"Effect verification failed: retry 2/3."
        }

        (2c) After third failed attempt, replan or adjust strategy:
        {
        "mode":"step",
        "plan_version":2,           // replanned
        "step_index":2,
        "next_action":{"name":"move_arm","args":{"object_bbox":[120,700,180,760],"arm":"left"}},  // example: re-approach
        "full_action_list":[ ... updated plan from this point ... ],
        "explanation":"Max retries reached for grasp; replanning with adjusted approach."
        }

        Final constraints:
        - Use TL xyxy absolute pixel integers exactly as provided.
        - Never normalize or fabricate bboxes.
        - Keep outputs minimal and immediately executable each turn.
        - Return ONLY a single valid JSON object with no code fences or prose.
        """.strip()
        )
    def parse_llava_response(self, llava_response: str | dict | list):
        """
        프롬프트 스키마:
        { "mode": "init"|"step", "plan_version": int, "step_index": int,
        "next_action": {"name": "...", "args": {...}} | None,
        "full_action_list": [ {"name": "...", "args": {...}}, ... ],
        "skip_log": [...], "visibility_warnings": [...], "explanation": "..." }
        """
        try:
            if isinstance(llava_response, str):
                m = re.search(r"\{[\s\S]*\}", llava_response)
                if not m:
                    return {}, []
                text = m.group(0)
                data = json.loads(text)
            elif isinstance(llava_response, dict):
                data = llava_response
            elif isinstance(llava_response, list) and len(llava_response) == 1:
                inner = llava_response[0]
                data = json.loads(inner) if isinstance(inner, str) else inner
            else:
                return {}, []

            next_action = data.get("next_action") or {}
            full_action_list = data.get("full_action_list", [])
            return next_action, full_action_list
        except Exception as e:
            print("Parsing error:", e)
            return {}, []

    def _normalize_actions(self, detection_targets, action_list):
        targets = set((detection_targets or []))
        fixed = []
        for a in action_list or []:
            try:
                name = a.get("name")
                args = dict(a.get("args", {}))
                if name in ("open_door", "close_door"):
                    obj = args.get("object")
                    if (obj == "door" or obj is None) and ("door-handle" in targets):
                        args["object"] = "door-handle"
                fixed.append({"name": name, "args": args})
            except Exception:
                fixed.append(a)
        return fixed


    def chat(self, user_command, img_path, detected: dict, feedback = None):
        feedback = feedback or {}

        has_image = False
        user_msg = {
            "role": "user",
            "content": (
                "USER_COMMAND: " + user_command + "\n" +
                "DETECTIONS: " + json.dumps(detected, ensure_ascii=False) + "\n" +
                "FEEDBACK: " + json.dumps(feedback, ensure_ascii=False)
            )
        }

        if img_path and isinstance(img_path, str) and img_path.strip():
            if os.path.exists(img_path):
                user_msg["images"] = [img_path]
                has_image = True
            else:
                print(f"[Warning] Image path not found: {img_path}. Proceeding without image.")
        if self.require_image and not has_image:
            result_json = {
                "mode": "step",
                "plan_version": 0,
                "step_index": 0,
                "next_action": None,
                "full_action_list": [],
                "explanation": "image_required"
            }
            self.messages.append(user_msg)
            raw_text = json.dumps(result_json, ensure_ascii=False)
            return {}, [], raw_text

        send_messages = self.messages + [user_msg]
        res = ollama.chat(model=self.model, messages=send_messages)

        assistant_msg = res.get("message", {})
        assistant_text = assistant_msg.get("content", "")

        self.messages.append(user_msg)
        if "role" not in assistant_msg:
            assistant_msg["role"] = "assistant"
        self.messages.append(assistant_msg)

        next_action, full_action_list = self.parse_llava_response(assistant_text)

        if full_action_list:
            full_action_list = self._normalize_actions(list(detected.keys()), full_action_list)
        if next_action:
            next_action = self._normalize_actions(list(detected.keys()), [next_action])[0]

        return next_action, full_action_list, assistant_text

if __name__ == "__main__":

    planner = LLaVA_planner(model="llava:13b", require_image=True)
    atexit.register(lambda: planner.realsense_stop())

    cnt = 1
    while True:
        try:
            user_command = input("user command: ").strip()
            if user_command.lower() == "exit":
                break
            if user_command.lower() == "reset":
                planner.reset()
                cnt = 1
                continue

            frame_bgr, img_path_for_vlm = planner.capture_current_frame(save_dir="/tmp")

            detected = planner.yolo_detect_objects(frame_bgr=frame_bgr)

            next_action, acts, raw = planner.chat(
                user_command=user_command,
                img_path=img_path_for_vlm,
                detected=detected,
                feedback=None
            )

            print(f"TURN{cnt} next_action:", next_action)
            print(f"TURN{cnt} full_action_list:", acts)
            print(f"RAW{cnt}:", raw)
            cnt += 1

        except Exception as e:
            print("[Error]", e)

