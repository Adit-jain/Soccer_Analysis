"""
A script to convert soccernet calibration dataset into field points
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import json
import cv2
from typing import Dict, List, Tuple, Optional
import math

class LineIntersectionCalculator:
    """
    A class to calculate field keypoints from SoccerNet line endpoints by computing line intersections.
    """
    
    def __init__(self):
        self.field_keypoints = {}
        self.lines = {}
    
    def load_soccernet_data(self, json_path: str) -> Dict:
        """
        Load SoccerNet calibration data from JSON file.
        
        Args:
            json_path: Path to the SoccerNet JSON file
            
        Returns:
            Dictionary containing line endpoints
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.lines = data
        return data
    
    def normalize_coordinates(self, point: Dict[str, float], image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert normalized coordinates to pixel coordinates.
        
        Args:
            point: Dictionary with 'x' and 'y' normalized coordinates (0-1)
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        height, width = image_shape[:2]
        x = int(point['x'] * width)
        y = int(point['y'] * height)
        return x, y
    
    def line_intersection(self, line1: List[Dict], line2: List[Dict]) -> Optional[Tuple[float, float]]:
        """
        Calculate intersection point of two lines defined by their endpoints.
        
        Args:
            line1: List of 2 dictionaries with 'x', 'y' coordinates (normalized 0-1)
            line2: List of 2 dictionaries with 'x', 'y' coordinates (normalized 0-1)
            
        Returns:
            Tuple of (x, y) intersection coordinates in normalized form, or None if lines don't intersect
        """
        if len(line1) < 2 or len(line2) < 2:
            return None
        
        # Extract points
        x1, y1 = line1[0]['x'], line1[0]['y']
        x2, y2 = line1[1]['x'], line1[1]['y']
        x3, y3 = line2[0]['x'], line2[0]['y']
        x4, y4 = line2[1]['x'], line2[1]['y']
        
        # Calculate line intersection using parametric form
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Calculate intersection point
        x_intersect = x1 + t * (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)
        
        return x_intersect, y_intersect
    
    def extend_line(self, line: List[Dict], extension_factor: float = 2.0) -> List[Dict]:
        """
        Extend a line segment by a given factor to help find intersections.
        
        Args:
            line: List of 2 dictionaries with 'x', 'y' coordinates
            extension_factor: Factor by which to extend the line
            
        Returns:
            Extended line endpoints
        """
        if len(line) < 2:
            return line
        
        x1, y1 = line[0]['x'], line[0]['y']
        x2, y2 = line[1]['x'], line[1]['y']
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Extend the line
        new_x1 = x1 - dx * (extension_factor - 1) / 2
        new_y1 = y1 - dy * (extension_factor - 1) / 2
        new_x2 = x2 + dx * (extension_factor - 1) / 2
        new_y2 = y2 + dy * (extension_factor - 1) / 2
        
        return [{'x': new_x1, 'y': new_y1}, {'x': new_x2, 'y': new_y2}]
    
    def calculate_field_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the specific 30 field keypoints from line intersections.
        
        Returns:
            Dictionary of keypoint names and their normalized coordinates
        """
        keypoints = {}
        
        # Helper function to safely get line data
        def get_line(key):
            return self.lines.get(key, [])
        
        # Get all line data
        side_line_top = get_line('Side line top')
        side_line_bottom = get_line('Side line bottom')
        side_line_left = get_line('Side line left')
        side_line_right = get_line('Side line right')
        
        big_rect_left_top = get_line('Big rect. left top')
        big_rect_left_main = get_line('Big rect. left main')
        big_rect_left_bottom = get_line('Big rect. left bottom')
        big_rect_right_top = get_line('Big rect. right top')
        big_rect_right_main = get_line('Big rect. right main')
        big_rect_right_bottom = get_line('Big rect. right bottom')
        
        small_rect_left_top = get_line('Small rect. left top')
        small_rect_left_main = get_line('Small rect. left main')
        small_rect_left_bottom = get_line('Small rect. left bottom')
        small_rect_right_top = get_line('Small rect. right top')
        small_rect_right_main = get_line('Small rect. right main')
        small_rect_right_bottom = get_line('Small rect. right bottom')
        
        middle_line = get_line('Middle line')
        circle_central = get_line('Circle central')
        circle_left = get_line('Circle left')
        circle_right = get_line('Circle right')
        
        # LEFT SIDE KEYPOINTS (1-18)
        
        # 1. Side line Top left
        if side_line_top and side_line_left:
            intersection = self.line_intersection(side_line_top, side_line_left)
            if intersection:
                keypoints['1_sideline_top_left'] = intersection
        
        # 2. Big rect left top pt 1 (closer to boundary)
        if side_line_left and big_rect_left_top:
            intersection = self.line_intersection(side_line_left, big_rect_left_top)
            if intersection:
                keypoints['2_big_rect_left_top_pt1'] = intersection
        
        # 3. Big rect left top pt 2
        if big_rect_left_top and big_rect_left_main:
            intersection = self.line_intersection(big_rect_left_top, big_rect_left_main)
            if intersection:
                keypoints['3_big_rect_left_top_pt2'] = intersection
        
        # 4. Big rect left bottom pt 1 (closer to boundary)
        if side_line_left and big_rect_left_bottom:
            intersection = self.line_intersection(side_line_left, big_rect_left_bottom)
            if intersection:
                keypoints['4_big_rect_left_bottom_pt1'] = intersection
        
        # 5. Big rect left bottom pt 2
        if big_rect_left_bottom and big_rect_left_main:
            intersection = self.line_intersection(big_rect_left_bottom, big_rect_left_main)
            if intersection:
                keypoints['5_big_rect_left_bottom_pt2'] = intersection
        
        # 6. Small rect left top pt 1 (closer to boundary)
        if side_line_left and small_rect_left_top:
            intersection = self.line_intersection(side_line_left, small_rect_left_top)
            if intersection:
                keypoints['6_small_rect_left_top_pt1'] = intersection
        
        # 7. Small rect left top pt 2
        if small_rect_left_top and small_rect_left_main:
            intersection = self.line_intersection(small_rect_left_top, small_rect_left_main)
            if intersection:
                keypoints['7_small_rect_left_top_pt2'] = intersection
        
        # 8. Small rect left bottom pt 1 (closer to boundary)
        if side_line_left and small_rect_left_bottom:
            intersection = self.line_intersection(side_line_left, small_rect_left_bottom)
            if intersection:
                keypoints['8_small_rect_left_bottom_pt1'] = intersection
        
        # 9. Small rect left bottom pt 2
        if small_rect_left_bottom and small_rect_left_main:
            intersection = self.line_intersection(small_rect_left_bottom, small_rect_left_main)
            if intersection:
                keypoints['9_small_rect_left_bottom_pt2'] = intersection
        
        # 10. Side line Bottom Left
        if side_line_bottom and side_line_left:
            intersection = self.line_intersection(side_line_bottom, side_line_left)
            if intersection:
                keypoints['10_sideline_bottom_left'] = intersection
        
        # 11. Left semicircle (highest x coordinate)
        if circle_left:
            # Find the point with highest x coordinate in the left circle
            max_x_point = max(circle_left, key=lambda p: p['x'])
            keypoints['11_left_semicircle_right'] = (max_x_point['x'], max_x_point['y'])
        
        # CENTER KEYPOINTS (12-18)
        
        # 12. Center line top
        if middle_line and side_line_top:
            intersection = self.line_intersection(middle_line, side_line_top)
            if intersection:
                keypoints['12_center_line_top'] = intersection
        
        # 13. Center line bottom
        if middle_line and side_line_bottom:
            intersection = self.line_intersection(middle_line, side_line_bottom)
            if intersection:
                keypoints['13_center_line_bottom'] = intersection
        
        # 14. Center circle left (lowest x coordinate)
        if circle_central:
            # Find the point with lowest x coordinate in the central circle
            min_x_point = min(circle_central, key=lambda p: p['x'])
            keypoints['14_center_circle_left'] = (min_x_point['x'], min_x_point['y'])
        
        # 15. Center circle top (intersects center line, higher y)
        if circle_central and middle_line:
            # For circle-line intersection, we need to find points on circle closest to the line
            # Simplified approach: find circle points with x closest to center line x
            if middle_line and len(middle_line) >= 2:
                center_line_x = (middle_line[0]['x'] + middle_line[1]['x']) / 2
                closest_points = sorted(circle_central, key=lambda p: abs(p['x'] - center_line_x))[:2]
                if len(closest_points) >= 2:
                    # Take the one with higher y (top)
                    top_point = max(closest_points, key=lambda p: p['y'])
                    keypoints['15_center_circle_top'] = (top_point['x'], top_point['y'])
        
        # 16. Center circle right (highest x coordinate)
        if circle_central:
            # Find the point with highest x coordinate in the central circle
            max_x_point = max(circle_central, key=lambda p: p['x'])
            keypoints['16_center_circle_right'] = (max_x_point['x'], max_x_point['y'])
        
        # 17. Center circle bottom (intersects center line, lower y)
        if circle_central and middle_line:
            # Similar to top, but take the one with lower y
            if middle_line and len(middle_line) >= 2:
                center_line_x = (middle_line[0]['x'] + middle_line[1]['x']) / 2
                closest_points = sorted(circle_central, key=lambda p: abs(p['x'] - center_line_x))[:2]
                if len(closest_points) >= 2:
                    # Take the one with lower y (bottom)
                    bottom_point = min(closest_points, key=lambda p: p['y'])
                    keypoints['17_center_circle_bottom'] = (bottom_point['x'], bottom_point['y'])
        
        # 18. Center of the football field
        if middle_line and len(middle_line) >= 2:
            # Calculate center point of the field (intersection of center line with field center)
            # Use midpoint of middle line or calculate intersection with horizontal center
            center_x = (middle_line[0]['x'] + middle_line[1]['x']) / 2
            center_y = (middle_line[0]['y'] + middle_line[1]['y']) / 2
            keypoints['18_field_center'] = (center_x, center_y)
        
        # RIGHT SIDE KEYPOINTS (19-30) - Mirror of left side
        
        # 19. Side line Top right (mirror of 1)
        if side_line_top and side_line_right:
            intersection = self.line_intersection(side_line_top, side_line_right)
            if intersection:
                keypoints['19_sideline_top_right'] = intersection
        
        # 20. Big rect right top pt 1 (closer to boundary) (mirror of 2)
        if side_line_right and big_rect_right_top:
            intersection = self.line_intersection(side_line_right, big_rect_right_top)
            if intersection:
                keypoints['20_big_rect_right_top_pt1'] = intersection
        
        # 21. Big rect right top pt 2 (mirror of 3)
        if big_rect_right_top and big_rect_right_main:
            intersection = self.line_intersection(big_rect_right_top, big_rect_right_main)
            if intersection:
                keypoints['21_big_rect_right_top_pt2'] = intersection
        
        # 22. Big rect right bottom pt 1 (closer to boundary) (mirror of 4)
        if side_line_right and big_rect_right_bottom:
            intersection = self.line_intersection(side_line_right, big_rect_right_bottom)
            if intersection:
                keypoints['22_big_rect_right_bottom_pt1'] = intersection
        
        # 23. Big rect right bottom pt 2 (mirror of 5)
        if big_rect_right_bottom and big_rect_right_main:
            intersection = self.line_intersection(big_rect_right_bottom, big_rect_right_main)
            if intersection:
                keypoints['23_big_rect_right_bottom_pt2'] = intersection
        
        # 24. Small rect right top pt 1 (closer to boundary) (mirror of 6)
        if side_line_right and small_rect_right_top:
            intersection = self.line_intersection(side_line_right, small_rect_right_top)
            if intersection:
                keypoints['24_small_rect_right_top_pt1'] = intersection
        
        # 25. Small rect right top pt 2 (mirror of 7)
        if small_rect_right_top and small_rect_right_main:
            intersection = self.line_intersection(small_rect_right_top, small_rect_right_main)
            if intersection:
                keypoints['25_small_rect_right_top_pt2'] = intersection
        
        # 26. Small rect right bottom pt 1 (closer to boundary) (mirror of 8)
        if side_line_right and small_rect_right_bottom:
            intersection = self.line_intersection(side_line_right, small_rect_right_bottom)
            if intersection:
                keypoints['26_small_rect_right_bottom_pt1'] = intersection
        
        # 27. Small rect right bottom pt 2 (mirror of 9)
        if small_rect_right_bottom and small_rect_right_main:
            intersection = self.line_intersection(small_rect_right_bottom, small_rect_right_main)
            if intersection:
                keypoints['27_small_rect_right_bottom_pt2'] = intersection
        
        # 28. Side line Bottom Right (mirror of 10)
        if side_line_bottom and side_line_right:
            intersection = self.line_intersection(side_line_bottom, side_line_right)
            if intersection:
                keypoints['28_sideline_bottom_right'] = intersection
        
        # 29. Right semicircle (lowest x coordinate) (mirror of 11)
        if circle_right:
            # Find the point with lowest x coordinate in the right circle
            min_x_point = min(circle_right, key=lambda p: p['x'])
            keypoints['29_right_semicircle_left'] = (min_x_point['x'], min_x_point['y'])
        
        self.field_keypoints = keypoints
        return self.field_keypoints, self.lines
    
    def visualize_keypoints(self, image_path: str, keypoints: Dict = None, lines: Dict = None, output_path: str = None):
        """
        Visualize the calculated keypoints and original lines on the image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated image (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        height, width = image.shape[:2]
        
        # Draw original lines
        if lines is not None:
            for line_name, line_points in lines.items():
                if len(line_points) >= 2 and line_name not in ['Circle left', 'Circle right']:  # Skip circle for now
                    pt1 = self.normalize_coordinates(line_points[0], image.shape)
                    pt2 = self.normalize_coordinates(line_points[1], image.shape)
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Green lines
                    cv2.putText(image, line_name[:10], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw circle points
            if 'Circle left' in lines:
                for point in lines['Circle left']:
                    pt = self.normalize_coordinates(point, image.shape)
                    cv2.circle(image, pt, 3, (0, 255, 0), -1)

            if 'Circle right' in lines:
                for point in lines['Circle right']:
                    pt = self.normalize_coordinates(point, image.shape)
                    cv2.circle(image, pt, 3, (0, 255, 0), -1)
        
        # Draw calculated keypoints
        if keypoints is not None:
            for keypoint_name, (x, y) in keypoints.items():
                pt = (int(x * width), int(y * height))
                cv2.circle(image, pt, 8, (0, 0, 255), -1)  # Red circles for keypoints
                cv2.putText(image, keypoint_name, (pt[0] + 10, pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save or show image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
        else:
            cv2.imshow('Field Keypoints', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()