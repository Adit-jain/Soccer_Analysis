
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
from ultralytics import YOLO
from player_detection import load_detection_model, get_detections
from player_tracking.tracking_constants import model_path, test_video
from player_tracking.assign_players import init_assignment_models, train_umap_kmeans, get_cluster_labels
from player_tracking.track_players import init_tracking_models, update_player_detections, annotate_players
import cv2
import time
from tqdm import tqdm
from utils import read_video, write_video
import pandas as pd
import pickle


#----------------------# Callback Functions #-----------------------#
def detection_callback(frame: np.ndarray, detection_model, siglip_model, siglip_processor, reducer, cluster_model) -> np.ndarray:
    """
    A callback function to detect players and assign teams.
    """

    # Get Detections
    detection_time = time.time()
    player_detections, ball_detections, referee_detections = get_detections(detection_model, frame)
    detection_time = time.time() - detection_time
    print(f"Detection Time: {detection_time:.2f}s")
    print(player_detections)
    print(ball_detections)
    print(referee_detections)

    return player_detections, ball_detections, referee_detections

def tracking_callback(player_detections, tracker):
    """
    A callback function to update player tracking.
    """
    
    # Update Player Tracking
    if len(player_detections.xyxy) > 0:
        player_detections = update_player_detections(player_detections, tracker)
    return player_detections

def cluster_callback(frame: np.ndarray, player_detections, siglip_model, siglip_processor, reducer, cluster_model) -> np.ndarray:
    """
    A callback function to cluster player embeddings and assign teams.
    """
        
    # Get Team Assignments
    assignment_time = time.time()
    clustered_embeddings = get_cluster_labels(frame, player_detections, siglip_model, siglip_processor, reducer, cluster_model)
    assignment_time = time.time() - assignment_time
    print(clustered_embeddings)
    print(f"Assignment Time: {assignment_time:.2f}s")
    
    # Assign Teams
    player_detections.class_id = clustered_embeddings

    return player_detections

#----------------------# Processing Functions #-----------------------#
def get_tracks(frames, detection_model, siglip_model, siglip_processor, reducer, cluster_model, tracker):
    """
    This function processes the video frames and gets the tracks of players, ball, and referee.
    It uses the detection and tracking callbacks to update the tracks.
    """
    # Loop through the video frames and get tracks
    tracks = {
        'player': {},
        'ball': {},
        'referee': {},
    }
    for index, frame in tqdm(enumerate(frames)):
        # Detect players and assign teams
        player_detections, ball_detections, referee_detections = detection_callback(frame, detection_model, siglip_model, siglip_processor, reducer, cluster_model)

        # Update player tracking
        player_detections = tracking_callback(player_detections, tracker)
        
        # Player tracking
        if len(player_detections.xyxy) > 0:
            for tracker_id, bbox in zip(player_detections.tracker_id, player_detections.xyxy):
                if index not in tracks['player']:
                    tracks['player'][index] = {}
                tracks['player'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['player'][index] = {-1:[None]*4}  # No players detected, assign -1 tracker_id with None bbox

        # Ball tracking
        if len(ball_detections.xyxy) > 0:
            for bbox in ball_detections.xyxy:
                tracks['ball'][index] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['ball'][index] = [None]*4  # No ball detected, assign None bbox

        # Referee tracking
        if len(referee_detections.xyxy) > 0:
            for tracker_id, bbox in zip(np.arange(len(referee_detections.xyxy)), referee_detections.xyxy):
                if index not in tracks['referee']:
                    tracks['referee'][index] = {}
                tracks['referee'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['referee'][index] = {-1:[None]*4}  # No referee detected, assign -1 tracker_id with None bbox

    return tracks
    

def interpolate_ball_tracks(tracks):
    """
    This function interpolates the ball tracks to fill in missing frames.
    """
    # Get the ball tracks
    ball_tracks = tracks['ball']
    
    df = pd.DataFrame.from_dict(ball_tracks, orient='index')
    df.columns = ['x1', 'y1', 'x2', 'y2']
    df = df.interpolate(method='linear', limit_direction='both', limit=30)
    
    # Get interpolated tracks
    new_tracks = {}
    for i, box in enumerate(df.to_numpy()):
        new_tracks[i] = box

    tracks['ball'] = new_tracks
    return tracks


def convert_to_supervision_detections(player_detections, ball_detections, referee_detections) -> list:
    """
    Annotate the frames with the player and ball tracks
    """

    # Get the player detections
    if player_detections is not None:
        player_detections = sv.Detections(
            xyxy=np.array(list(player_detections.values())),
            class_id=np.array([0] * len(player_detections)),
            tracker_id=np.array(list(player_detections.keys()))
        )

    # Get the ball detections
    if ball_detections is not None:
        ball_detections = sv.Detections(
            xyxy=np.array([ball_detections]),
            class_id=np.array([2]),
        )

    # Get the player detections
    if referee_detections is not None:
        referee_detections = sv.Detections(
            xyxy=np.array(list(referee_detections.values())),
            class_id=np.array([3] * len(referee_detections)),
            tracker_id=np.array(list(referee_detections.keys()))
        )

    return player_detections, ball_detections, referee_detections


def annotate_frames(frames, tracks, ellipse_annotator, triangle_annotator, label_annotator, siglip_model, siglip_processor, reducer, cluster_model):
    """
    This function annotates the frames with the player and ball tracks.
    It uses the Supervision library to create detections and annotate the frames.
    """
    annotated_frames = []
    for index, frame in enumerate(frames):
        
        # Get the player, ball, and referee detections
        player_detections = tracks['player'][index]
        ball_detections = tracks['ball'][index]
        referee_detections = tracks['referee'][index]
        if -1 in player_detections:
            player_detections = None
        if -1 in referee_detections:
            referee_detections = None
        if (not all(ball_detections)) or np.isnan(ball_detections).all():
            ball_detections = None

        # Convert the tracks to detections
        player_detections, ball_detections, referee_detections = convert_to_supervision_detections(player_detections, ball_detections, referee_detections)

        # Assign teams to players
        if player_detections is not None:
            player_detections = cluster_callback(frame, player_detections, siglip_model, siglip_processor, reducer, cluster_model)

        # Annotate the players
        annotated_frame = annotate_players(frame, player_detections, ball_detections, referee_detections, ellipse_annotator, triangle_annotator, label_annotator)
        annotated_frames.append(annotated_frame)

    return annotated_frames

#----------------------# Driver Functions #-----------------------#
def track_players(video_path, model_path):
    """
    This function detects players in a video in real-time
    """
    # Load the Model
    model_init_time = time.time()
    print("Initializing the model...")
    detection_model = load_detection_model(model_path)

    # Initialize the annotators
    print("Initializing the annotators...")
    tracker, ellipse_annotator, traingle_annotator, label_annotator = init_tracking_models()

    # Initialize the Player Assignment Models
    print("Initializing the player assignment models...")
    siglip_model, siglip_processor, reducer, cluster_model = init_assignment_models()
    model_init_time = time.time() - model_init_time
    print(f"Model Initialization Time: {model_init_time:.2f}s")

    # Train the UMAP and KMeans models
    print("Training the UMAP and KMeans models...")
    training_kmeans_time = time.time()
    _, reducer, cluster_model = train_umap_kmeans(video_path, detection_model, siglip_model, siglip_processor, reducer, cluster_model)
    training_kmeans_time = time.time() - training_kmeans_time
    print(f"Training KMeans Time: {training_kmeans_time:.2f}s")

    # Read the video
    print("Reading the video...")
    frames = read_video(video_path)  # Read all frames from the video

    # Get tracks for each frame
    print("Getting tracks for each frame...")
    tracks = get_tracks(frames, detection_model, siglip_model, siglip_processor, reducer, cluster_model, tracker)

    # Interpolate the ball tracks
    print("Interpolating the ball tracks...")
    tracks = interpolate_ball_tracks(tracks)

    # Annotate the frame
    print("Annotating the frames...")
    annotated_frames = annotate_frames(frames, tracks, ellipse_annotator, traingle_annotator, label_annotator, siglip_model, siglip_processor, reducer, cluster_model)

    # Initialize the video writer
    output_path = video_path.replace(".mp4", "_tracked.mp4")
    write_video(annotated_frames, output_path, fps=30)

if __name__ == "__main__":
    track_players(test_video, model_path)