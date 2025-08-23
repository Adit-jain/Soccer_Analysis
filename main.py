import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline
from constants import model_path, test_video


def track_players(video_path, model_path):
    """
    Main function to track players in a soccer video.
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO detection model
    """
    # Initialize pipelines
    tracking_pipeline = TrackingPipeline(model_path)
    processing_pipeline = ProcessingPipeline()
    
    # Initialize all models
    tracking_pipeline.initialize_models()
    
    # Train team assignment models
    tracking_pipeline.train_team_assignment_models(video_path)
    
    # Read video frames
    frames = processing_pipeline.read_video_frames(video_path, frame_count=-1)
    
    # Get tracks for each frame
    tracks = tracking_pipeline.get_tracks(frames)
    
    # Interpolate ball tracks
    tracks = processing_pipeline.interpolate_ball_tracks(tracks)
    
    # Annotate frames
    annotated_frames = tracking_pipeline.annotate_frames(frames, tracks)
    
    # Write output video
    output_path = processing_pipeline.generate_output_path(video_path)
    processing_pipeline.write_video_output(annotated_frames, output_path, fps=30)
    
    print(f"Tracking completed! Output saved to: {output_path}")


if __name__ == "__main__":
    track_players(test_video, model_path)