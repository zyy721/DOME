import os
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(folder_path, max_height=480):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                mp4_path = os.path.join(root, file)
                gif_path = os.path.join(root, os.path.splitext(file)[0] + ".gif")

                print(f"Converting: {mp4_path} -> {gif_path}")

                try:
                    clip = VideoFileClip(mp4_path)
                    # Resize the clip to reduce dimensions
                    if clip.h > max_height:
                        clip = clip.resize(height=max_height)
                    clip.write_gif(gif_path)
                    print(f"Saved: {gif_path}")
                except Exception as e:
                    print(f"Failed to convert {mp4_path}: {e}")

if __name__ == "__main__":
    folder_to_traverse = '/home/bj-01/dome_github/static/videos'
    if os.path.exists(folder_to_traverse):
        convert_mp4_to_gif(folder_to_traverse)
    else:
        print("Invalid folder path!")
