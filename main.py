import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video
from keypoint_detection.keypoint_constants import keypoint_model_path


def run_tracking_analysis(video_path: str, detection_model_path: str):
    """Run complete tracking analysis with team assignment."""
    tracking_pipeline = TrackingPipeline(detection_model_path)
    processing_pipeline = ProcessingPipeline()
    
    tracking_pipeline.initialize_models()
    tracking_pipeline.train_team_assignment_models(video_path)
    
    frames = processing_pipeline.read_video_frames(video_path, frame_count=-1)
    tracks = tracking_pipeline.get_tracks(frames)
    tracks = processing_pipeline.interpolate_ball_tracks(tracks)
    
    annotated_frames = tracking_pipeline.annotate_frames(frames, tracks)
    output_path = processing_pipeline.generate_output_path(video_path, "_tracked")
    processing_pipeline.write_video_output(annotated_frames, output_path, fps=30)
    
    print(f"Tracking analysis completed! Output saved to: {output_path}")
    return output_path


def run_tactical_analysis(video_path: str, keypoint_model_path: str, detection_model_path: str, 
                         create_overlay: bool = True, frame_count: int = 300):
    """Run tactical analysis with field coordinate transformation."""
    tactical_pipeline = TacticalPipeline(keypoint_model_path, detection_model_path)
    processing_pipeline = ProcessingPipeline()
    
    suffix = "_tactical_overlay" if create_overlay else "_tactical_only"
    output_path = processing_pipeline.generate_output_path(video_path, suffix)
    
    tactical_pipeline.analyze_video(video_path, output_path, frame_count, create_overlay)
    
    print(f"Tactical analysis completed! Output saved to: {output_path}")
    return output_path


def run_detection_analysis(video_path: str, detection_model_path: str, frame_count: int = 300):
    """Run basic detection analysis."""
    detection_pipeline = DetectionPipeline(detection_model_path)
    processing_pipeline = ProcessingPipeline()
    
    output_path = processing_pipeline.generate_output_path(video_path, "_detected")
    detection_pipeline.detect_in_video(video_path, output_path, frame_count)
    
    print(f"Detection analysis completed! Output saved to: {output_path}")
    return output_path


def run_keypoint_analysis(video_path: str, keypoint_model_path: str, frame_count: int = 300):
    """Run keypoint detection analysis."""
    keypoint_pipeline = KeypointPipeline(keypoint_model_path)
    processing_pipeline = ProcessingPipeline()
    
    output_path = processing_pipeline.generate_output_path(video_path, "_keypoints")
    keypoint_pipeline.detect_in_video(video_path, output_path, frame_count)
    
    print(f"Keypoint analysis completed! Output saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Choose analysis type (uncomment desired option)
    
    # Option 1: Tactical Analysis with Overlay (Recommended)
    run_tactical_analysis(test_video, keypoint_model_path, model_path, create_overlay=True, frame_count=300)
    
    # Option 2: Complete Tracking Analysis with Team Assignment
    # run_tracking_analysis(test_video, model_path)
    
    # Option 3: Basic Detection Analysis
    # run_detection_analysis(test_video, model_path, frame_count=300)
    
    # Option 4: Keypoint Detection Analysis
    # run_keypoint_analysis(test_video, keypoint_model_path, frame_count=300)
    
    # Option 5: Tactical Analysis without Overlay (Tactical View Only)
    # run_tactical_analysis(test_video, keypoint_model_path, model_path, create_overlay=False, frame_count=300)