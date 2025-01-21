import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
import argparse

# Timing array
timings = [0.0000, 7.2875, 8.1951, 8.4696, 8.7700, 9.6976, 10.5730, 10.8471, 11.0900, 11.7600, 11.8700, 12.0800, 12.3839, 13.0600, 13.1500, 13.3200, 13.5800, 14.1966, 14.4780, 14.7800, 15.3700, 15.6755, 16.8067, 17.6300, 17.9177, 18.2200, 19.1693, 19.9900, 20.2816, 20.5557, 21.2600, 21.4100, 21.5600, 21.7700, 22.4900, 22.6900, 22.8200, 23.1100, 23.6900, 23.9542, 24.2900, 24.6000, 25.1200, 26.4948, 27.6277, 28.2383, 28.7599, 29.0495, 29.3221, 30.1089, 30.4385, 31.0247, 32.0809, 33.3019, 33.4287, 33.5917, 33.8612, 34.4300, 34.6165, 34.7677, 35.0214, 35.5200, 35.6400, 35.7900, 36.0800, 36.6500, 36.7956, 36.9470, 37.1992]



def create_video():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_output = os.path.join(script_dir, 'temp_output.mp4')
    final_output = os.path.join(script_dir, 'final_output.mp4')
    
    # Get all images from the directory
    image_dir = 'files/channel_1315381196221321258'
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Only take the number of images we need based on timing intervals
    needed_images = len(timings) - 1  # -1 because timings include start time
    images = all_images[:needed_images]
    
    if len(all_images) > needed_images:
        print(f"Note: Using first {needed_images} images out of {len(all_images)} found images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, images[0]))
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Make sure this matches your audio's timing
    output_video = 'temp_output.mp4'
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Calculate exact frame positions
    frame_positions = [round(time * fps) for time in timings]
    audio = AudioFileClip('files/123.mp3')
    total_frames = round(audio.duration * fps)
    current_frame = 0
    
    print("\nDebug timing information:")
    print(f"Audio duration: {audio.duration:.4f}s")
    print(f"Last timing point: {timings[-1]:.4f}s")
    print(f"Second to last timing point: {timings[-2]:.4f}s")
    print(f"Time difference for last frame: {timings[-1] - timings[-2]:.4f}s\n")
    
    # Process each image except the last one
    for i, image_file in enumerate(images):
        img = cv2.imread(os.path.join(image_dir, image_file))
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # Use the actual timestamps instead of frame calculations
        start_time = timings[i]
        start_frame = round(start_time * fps)
        
        if i < len(images):
            end_time = timings[i + 1]
            end_frame = round(end_time * fps)
            print(f"\nImage {i+1} ({image_file}):")
            print(f"- Starting at: {start_time:.4f}s (frame {start_frame})")
            print(f"- Ending at: {end_time:.4f}s (frame {end_frame})")
        else:
            # Final extension
            end_frame = total_frames
            extension_time = (total_frames/fps) - start_time
            print(f"\nFinal Image Extension Details ({image_file}):")
            print(f"- Starting from: {start_time:.4f}s (frame {start_frame})")
            print(f"- Extending to: {end_frame/fps:.4f}s (frame {end_frame})")
            print(f"- Extension duration: {extension_time:.4f}s")
            
            # Add progress updates for EVERY frame
            frames_written = 0
            n_frames = end_frame - start_frame
            
            print("\nWriting extended frames:")
            for frame_num in range(n_frames):  # Remove the +1 and use direct frame counting
                out.write(img)
                frames_written += 1
                current_frame += 1
                
                # Show progress for EVERY SINGLE FRAME, no exceptions
                current_time = start_frame/fps + (frames_written / fps)
                progress = (frames_written / n_frames) * 100
                print(f"Progress: {progress:.1f}% - Written frame {frames_written}/{n_frames} "
                      f"(Time: {current_time:.4f}s) - Image: {image_file}")
            continue  # Skip the regular frame writing loop for the last image

        n_frames = end_frame - start_frame
        for _ in range(n_frames):
            out.write(img)
            current_frame += 1
    
    # Handle the last image separately
    current_image_index = len(images) - 1
    if current_image_index < len(all_images) - 1:
        last_image_file = all_images[current_image_index + 1]  # Get next image from all_images
    else:
        last_image_file = images[-1]  # Fallback to current image if we're at the end
    last_img = cv2.imread(os.path.join(image_dir, last_image_file))
    if last_img.shape[:2] != (height, width):
        last_img = cv2.resize(last_img, (width, height))
    
    start_frame = frame_positions[-1]
    start_time = timings[-1]
    end_frame = total_frames
    end_time = audio.duration
    
    # Here's where we calculate n_frames
    n_frames = end_frame - start_frame + 1  # Add 1 to include both boundary frames
    
    actual_start_time = start_frame / fps
    actual_end_time = end_frame / fps
    extension_duration = actual_end_time - actual_start_time
    
    print(f"\nLast frame extension details:")
    print(f"File: {last_image_file}")
    print(f"Starting extension at: {actual_start_time:.4f}s (frame {start_frame})")
    print(f"Extending until: {actual_end_time:.4f}s (frame {end_frame})")
    print(f"Extension duration: {extension_duration:.4f}s ({n_frames} frames)")
    print(f"Current frame count: {current_frame}")
    
    # Write the final frame with progress updates
    print("\nWriting extended frames:")
    frames_written = 0
    update_interval = 1
    
    for i in range(n_frames + 1):  # Add +1 to match 1-based counting
        out.write(last_img)
        frames_written += 1
        current_frame += 1
        
        if frames_written % update_interval == 0:
            current_time = actual_start_time + (frames_written / fps)
            progress = (frames_written / n_frames) * 100
            print(f"Progress: {progress:.1f}% - Written frame {frames_written}/{n_frames} "
                  f"(Time: {current_time:.4f}s) - Image: {last_image_file}")
    
    print(f"\nExtension complete - Final frame count: {current_frame}")
    
    out.release()
    audio.close()
    
    print(f"\nFinal frame information:")
    print(f"Total frames created: {current_frame}")
    print(f"Expected duration: {audio.duration:.4f}s")
    print(f"Actual duration: {current_frame/fps:.4f}s")
    
    # Add audio with exact duration matching
    video = VideoFileClip(output_video)
    audio = AudioFileClip('files/123.mp3')
    
    # Get audio duration and calculate additional frames needed
    audio_duration = audio.duration
    additional_duration = audio_duration - timings[-1]
    additional_frames = round(additional_duration * fps)
    
    print(f"\nOriginal video duration: {video.duration:.3f}s")
    print(f"Audio duration: {audio_duration:.3f}s")
    '''
    if additional_duration > 0:
        print(f"\nExtending last image for {additional_duration:.3f}s ({additional_frames} frames)")
        temp_extension = os.path.join(script_dir, 'temp_extension.mp4')
        temp_combined = os.path.join(script_dir, 'temp_combined.mp4')
        
        # Create a temporary file for the additional frames
        out = cv2.VideoWriter(temp_extension, fourcc, fps, (width, height))
        last_img = cv2.imread(os.path.join(image_dir, images[-1]))
        if last_img.shape[:2] != (height, width):
            last_img = cv2.resize(last_img, (width, height))
        
        # Add the last image for the remaining duration
        for _ in range(additional_frames):
            out.write(last_img)
        out.release()
        
        # Concatenate the videos using moviepy
        main_video = VideoFileClip(output_video)
        extension_video = VideoFileClip(temp_extension)
        
        print(f"Main video duration: {main_video.duration:.3f}s")
        print(f"Extension duration: {extension_video.duration:.3f}s")
        
        final_video = concatenate_videoclips([main_video, extension_video])
        
        # Write the combined video to a new temporary file
        final_video.write_videofile(temp_combined, codec='libx264', fps=fps)
        
        # Clean up and prepare for audio addition
        main_video.close()
        extension_video.close()
        video.close()
        
        if os.path.exists(temp_extension):
            os.remove(temp_extension)
            
        os.rename(temp_combined, output_video)
        
        # Reload the video for audio addition
        video = VideoFileClip(output_video)

    print(f"Final video duration before audio: {video.duration:.3f}s")
    '''
    # Combine video and audio
    final_video = video.set_audio(audio)
    final_video.write_videofile(final_output, codec='libx264', fps=fps, 
                               audio_codec='aac', audio_fps=44100)
    
    # Clean up
    video.close()
    audio.close()
    final_video.close()
    
    if os.path.exists(final_output):
        print(f"Video creation completed! Saved to: {final_output}")
    else:
        print("Error: Final video was not created!")

def apply_effects(input_video='final_output.mp4', output_video='final_output_with_effects.mp4'):
    print("Applying visual effects...")
    
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create absolute paths for all files
    input_video_path = os.path.join(script_dir, input_video)
    output_video_path = os.path.join(script_dir, output_video)
    temp_fade_video = os.path.join(script_dir, 'temp_fade.mp4')
    temp_zoom_video = os.path.join(script_dir, 'temp_zoom.mp4')
    
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found at: {input_video_path}")
    
    print(f"Processing video at: {input_video_path}")
    
    # Load the video with MoviePy for fade effect
    video = VideoFileClip(input_video_path)
    
    # Step 1: Apply fade in and save
    print("Applying fade effect...")
    faded = video.fadein(6.0)
    faded.write_videofile(temp_fade_video, codec='libx264', fps=video.fps)
    faded.close()
    video.close()
    
    # Step 2: Apply zoom effect using OpenCV
    print("Applying zoom effect...")
    cap = cv2.VideoCapture(temp_fade_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate center point once - using exact center pixels
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    
    # Create video writer with h264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_zoom_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:  # Changed from while cap.isOpened()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate zoom factor with smoother progression
        # Ensure we don't hit 1.0 progress until the very last frame
        progress = frame_count / (total_frames - 1) if total_frames > 1 else 1.0
        # Use cubic easing for smoother zoom
        smooth_progress = progress * progress * (3 - 2 * progress)
        zoom_factor = 1 + (0.15 * smooth_progress)  # 15% max zoom
        
        # Create transformation matrix
        M = np.float32([
            [zoom_factor, 0, center_x * (1 - zoom_factor)],
            [0, zoom_factor, center_y * (1 - zoom_factor)],
        ])
        
        # Apply affine transformation
        zoomed = cv2.warpAffine(frame, M, (width, height), 
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT)
        
        # Write frame
        out.write(zoomed)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
    
    # Verify frame count
    print(f"Processed {frame_count} frames out of {total_frames} total frames")
    
    # Release resources
    cap.release()
    out.release()
    
    # Combine with original audio
    print("Adding audio...")
    zoomed_video = VideoFileClip(temp_zoom_video)
    original_video = VideoFileClip(temp_fade_video)
    final_video = zoomed_video.set_audio(original_video.audio)
    
    # Write final video with high quality settings
    final_video.write_videofile(output_video_path, 
                              codec='libx264',
                              fps=fps,
                              audio_codec='aac',
                              audio_fps=44100,
                              preset='slow',
                              bitrate='8000k')
    
    # Clean up
    zoomed_video.close()
    original_video.close()
    final_video.close()
    
    if os.path.exists(temp_fade_video):
        os.remove(temp_fade_video)
    if os.path.exists(temp_zoom_video):
        os.remove(temp_zoom_video)
    
    print("Effects applied successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--just-effects', action='store_true', 
                       help='Only apply effects to existing final_output.mp4')
    args = parser.parse_args()
    
    if args.just_effects:
        apply_effects()
    else:
        create_video()
        print("\nApplying effects to the video...")
        apply_effects()