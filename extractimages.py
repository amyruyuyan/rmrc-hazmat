"""
Video Frame Extractor (for hazmat datasets)

Extracts every frame from a video file and saves them as individual images
to a specified output directory. (might make a new dataset for this just so 
the images don't get messed up and the pathing is better)
"""

import cv2
import os
import sys
from pathlib import Path


def extract_frames_from_video(video_path, output_folder, frame_prefix="frame"):
    """
    Extract all frames from a video and save them as images.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the folder where frames will be saved
        frame_prefix (str): Prefix for the frame filenames
    
    Returns:
        tuple: (success, total_frames, error_message)
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        return False, 0, f"Video file not found: {video_path}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        return False, 0, f"Error: Could not open video file {video_path}"
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video Information:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Output folder: {output_folder}")
    print()
    
    frame_count = 0
    success_count = 0
    
    # Extract frames
    print("Extracting frames...")
    try:
        while True:
            # Read frame
            ret, frame = video_capture.read()
            
            # Break if no more frames
            if not ret:
                break
            
            # Generate filename with zero-padded frame number
            frame_filename = f"{frame_prefix}_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            # Save frame as JPG image
            if cv2.imwrite(frame_path, frame):
                success_count += 1
                # Show progress every 100 frames
                if success_count % 100 == 0:
                    print(f"Processed {success_count} frames...")
            else:
                print(f"Warning: Failed to save frame {frame_count}")
            
            frame_count += 1
    
    except Exception as e:
        video_capture.release()
        return False, success_count, f"Error during frame extraction: {str(e)}"
    
    finally:
        video_capture.release()
    
    return True, success_count, None


def get_user_inputs():
    """Get video path and output folder from user input."""
    print("Video Frame Extractor")
    print("=" * 50)
    
    # Get video file path
    while True:
        video_path = input("Enter the path to your video file: ").strip()
        if video_path:
            # Remove quotes if present
            video_path = video_path.strip('"\'')
            if os.path.exists(video_path):
                break
            else:
                print(f"Error: Video file not found - {video_path}")
                print("Please check the path and try again.")
        else:
            print("Please enter a valid video file path.")
    
    # Get output folder path
    while True:
        output_folder = input("Enter the folder path where you want to save the images: ").strip()
        if output_folder:
            # Remove quotes if present
            output_folder = output_folder.strip('"\'')
            break
        else:
            print("Please enter a valid folder path.")
    
    # Get optional custom prefix for frame filenames
    frame_prefix = input("Enter a prefix for image filenames (default: 'frame'): ").strip()
    if not frame_prefix:
        frame_prefix = "frame"
    
    return video_path, output_folder, frame_prefix

def main():
    """Main function to run the frame extractor."""
    
    # Check if command line arguments are provided
    if len(sys.argv) == 3:
        video_path = sys.argv[1]
        output_folder = sys.argv[2]
        frame_prefix = "frame"
    elif len(sys.argv) == 4:
        video_path = sys.argv[1]
        output_folder = sys.argv[2]
        frame_prefix = sys.argv[3]
    else:
        # Interactive mode - get inputs from user
        video_path, output_folder, frame_prefix = get_user_inputs()
    
    print(f"\nStarting frame extraction...")
    print(f"Video: {video_path}")
    print(f"Output folder: {output_folder}")
    print(f"Image prefix: {frame_prefix}")
    print("-" * 50)
    
    # Extract frames from video
    success, frame_count, error = extract_frames_from_video(video_path, output_folder, frame_prefix)
    
    if success:
        print(f"\n✅ Success! Extracted {frame_count} frames")
        print(f"Images saved to: {output_folder}")
        
        # Show example filenames
        if frame_count > 0:
            print(f"\nExample files created:")
            print(f"  - {frame_prefix}_000000.jpg")
            if frame_count > 1:
                print(f"  - {frame_prefix}_000001.jpg")
            if frame_count > 2:
                print(f"  - ...")
                print(f"  - {frame_prefix}_{frame_count-1:06d}.jpg")
    else:
        print(f"\n Error: {error}")
        if frame_count > 0:
            print(f"Successfully extracted {frame_count} frames before error occurred.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)