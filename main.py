import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video
from keypoint_detection.keypoint_constants import keypoint_model_path
import numpy as np
import time
from tqdm import tqdm
import supervision as sv


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities."""
    
    def __init__(self, detection_model_path: str, keypoint_model_path: str):
        """Initialize all pipeline components.
        
        Args:
            detection_model_path: Path to YOLO detection model
            keypoint_model_path: Path to YOLO keypoint detection model
        """
        self.detection_pipeline = DetectionPipeline(detection_model_path)
        self.keypoint_pipeline = KeypointPipeline(keypoint_model_path)
        self.tracking_pipeline = TrackingPipeline(detection_model_path)
        self.tactical_pipeline = TacticalPipeline(keypoint_model_path, detection_model_path)
        self.processing_pipeline = ProcessingPipeline()
        
    def initialize_models(self):
        """Initialize all models required for complete analysis."""
        
        print("Initializing all pipeline models...")
        start_time = time.time()
        
        # Initialize all pipeline models
        self.detection_pipeline.initialize_model()
        self.keypoint_pipeline.initialize_model()
        self.tracking_pipeline.initialize_models()
        self.tactical_pipeline.initialize_models()
        
        init_time = time.time() - start_time
        print(f"All models initialized in {init_time:.2f}s")
        
    def analyze_video(self, video_path: str, frame_count: int = -1, output_suffix: str = "_complete_analysis"):
        """Run complete end-to-end soccer analysis.
        
        Flow:
        1. Read video
        2. Detect keypoints and objects (players, ball, referees)
        3. Update with tracking
        4. Tactical Analysis
        5. Interpolate ball tracks
        6. Assign Teams
        7. Tactical Overlay
        8. Save Video
        
        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file
            
        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        total_start_time = time.time()
        
        # Step 1: Initialize all models
        self.initialize_models()
        
        # Step 2: Train team assignment models
        print("\n[Step 2/8] Training team assignment models...")
        self.tracking_pipeline.train_team_assignment_models(video_path)
        
        # Step 3: Read video frames
        print("\n[Step 3/8] Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        print(f"Loaded {len(frames)} frames for processing")
        
        # Step 4: Process all frames with detections, tracking, and tactical analysis
        print("\n[Step 4/8] Processing frames with complete analysis...")
        tactical_frames = []
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}}
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):

            # Detect keypoints and objects
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
            
            # Update with tracking
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)

            # Team assignment
            player_detections, _ = self.tracking_pipeline.clustering_callback(frame, player_detections)
            
            # Store tracks for interpolation
            all_tracks = self.tracking_pipeline.convert_detection_to_tracks(player_detections, ball_detections, referee_detections, all_tracks, i)
            
            # Get tactical frame from detections
            tactical_frame, _ = self.tactical_pipeline.process_detections_for_tactical_analysis(player_detections, ball_detections, referee_detections, keypoints)
            tactical_frames.append(tactical_frame)

        # Step 5: Ball track interpolation
        print("\n[Step 5/8] Interpolating ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)
        
        # Step 6: Player Annotation
        print("\n[Step 6/8] Assigning teams and Annotating frames with detections...")
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)

        # Step 7: Overlay object annotated frames with tactical frames
        print("\n[Step 7/8] Overlaying Tactical frames...")
        output_frames = []
        assert len(object_annotated_frames) == len(tactical_frames)
        for o_frame, t_frame in zip(object_annotated_frames, tactical_frames):
            output_frame = self.tactical_pipeline.create_overlay_frame(o_frame, t_frame, overlay_size=(500, 350))
            output_frames.append(output_frame)

        # Step 8: Write final output video
        print("\n[Step 8/8] Writing complete analysis video...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path)
        
        # Summary
        total_time = time.time() - total_start_time
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        print(f"Output saved to: {output_path}")
        
        return output_path



if __name__ == "__main__":
    # Run Complete End-to-End Soccer Analysis Pipeline
    print("Starting Soccer Analysis...")
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
    output_video = pipeline.analyze_video(test_video, frame_count=-1)    
    print(f"\nAnalysis finished! Output video: {output_video}")