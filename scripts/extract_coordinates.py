"""
extract_coordinates.py

RADAR 파이프라인과 동일하게 영상을 처리하되, 화면에 그리는 대신
프레임별 선수 위치를 실제 피치 좌표(cm)로 변환해 CSV로 저장한다.

출력 컬럼: frame, tracker_id, role, team, x, y
  - role : player / goalkeeper / referee
  - team : 0 또는 1 (선수·골키퍼), 심판은 빈 값
  - x, y : 피치 좌표 (단위 cm, 경기장 12000 x 7000)

사용 예:
  python extract_coordinates.py \
    --source_video_path data/2e57b9_0.mp4 \
    --output_csv_path data/coordinates.csv \
    --device mps
"""

import argparse
import csv
import os
from typing import List

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

ROLE_BY_CLASS = {
    PLAYER_CLASS_ID: 'player',
    GOALKEEPER_CLASS_ID: 'goalkeeper',
    REFEREE_CLASS_ID: 'referee',
}


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def extract(source_video_path: str, output_csv_path: str, device: str) -> None:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # 1) 팀 분류기 학습용 크롭 수집 (영상 전체를 STRIDE 간격으로 훑음)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # 2) 본 처리 패스 (모든 프레임) — 좌표를 CSV로 기록
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    skipped_frames = 0
    written_rows = 0

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'tracker_id', 'role', 'team', 'x', 'y'])

        for frame_idx, frame in enumerate(tqdm(frame_generator, desc='extracting')):
            # 피치 키포인트 (호모그래피용)
            result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            # 선수/골키퍼/심판 검출 + 추적
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            # 팀 분류 (선수가 있어야 가능)
            if len(players) == 0:
                skipped_frames += 1
                continue
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)

            if len(goalkeepers) > 0:
                goalkeepers_team_id = resolve_goalkeepers_team_id(
                    players, players_team_id, goalkeepers)
            else:
                goalkeepers_team_id = np.array([], dtype=int)

            merged = sv.Detections.merge([players, goalkeepers, referees])
            team_lookup = (
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [None] * len(referees)
            )

            # 호모그래피: 키포인트가 4개 미만이면 변환 불가 → 프레임 스킵
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.sum() < 4:
                skipped_frames += 1
                continue
            try:
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                )
                xy = merged.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_xy = transformer.transform_points(points=xy)
            except Exception:
                skipped_frames += 1
                continue

            for i in range(len(merged)):
                class_id = int(merged.class_id[i])
                role = ROLE_BY_CLASS.get(class_id, str(class_id))
                team = team_lookup[i]
                tracker_id = merged.tracker_id[i] if merged.tracker_id is not None else None
                x, y = transformed_xy[i]
                writer.writerow([
                    frame_idx,
                    '' if tracker_id is None else int(tracker_id),
                    role,
                    '' if team is None else int(team),
                    round(float(x), 1),
                    round(float(y), 1),
                ])
                written_rows += 1

    print(f'완료: {output_csv_path}')
    print(f'기록된 행: {written_rows}, 스킵된 프레임: {skipped_frames}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract pitch coordinates to CSV.')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--output_csv_path', type=str, default='data/coordinates.csv')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    extract(
        source_video_path=args.source_video_path,
        output_csv_path=args.output_csv_path,
        device=args.device,
    )
