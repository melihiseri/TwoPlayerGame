import os
import time
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from TwoPlayerGame.game import Universe
from TwoPlayerGame.utils import set_seed
from TwoPlayerGame.visualization import plot_all_actions, plot_cumulative_rewards, plot_moving_average_rewards


def manual_animation(N_game: int, n_experiment: int, save: bool = False, filename: str = "animation.mp4"):
    """
    Runs the game for N_game steps, updates plots dynamically, and optionally saves an animation.

    Args:
        N_game (int): Number of game rounds.
        save (bool): If True, saves frames and creates an animation.
        filename (str): Filename for the saved animation.
    """
    from TwoPlayerGame.game import Universe

    experiments = [
        (0.3, -0.1),
        (0.05, 0.1),
        (1.0, -1.0),
        (1.0, 0.0),
    ]
    
    Uni = Universe(*experiments[n_experiment])

    os.makedirs("frames", exist_ok=True)
    frames = []

    start_time = time.time()
    for frame in range(N_game):
        Uni.play_the_game()
        if save:
            Uni.update()
            Uni.fig.canvas.draw()
            frame_filename = f"frames/frame_{frame:03d}.png"
            plt.savefig(frame_filename, dpi=150)
            frames.append(frame_filename)

        # --- Display progress
        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / (frame + 1)
        remaining_time = avg_time_per_frame * (N_game - frame - 1)
        print(f"Frame {frame + 1}/{N_game} | Elapsed: {elapsed_time:.2f}s | Estimated Left: {remaining_time:.2f}s | Last P1 action: {Uni.a1_past[-1]}", end='\r')

    if save:
        if shutil.which("ffmpeg") is None:
            print("\nError: FFmpeg not found! Install it using 'sudo apt install ffmpeg' or 'brew install ffmpeg'.")
            return

        # --- H.264 Better Compatibility
        ffmpeg_command = f"ffmpeg -r 4 -i frames/frame_%03d.png -vf \"scale=1920:1080:flags=lanczos\" -vcodec libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart animation_{filename}.mp4"
        # --- H.265 Better Compression
        # ffmpeg_command = f"ffmpeg -r 4 -i frames/frame_%03d.png -vf \"scale=1920:1080:flags=lanczos\" -vcodec libx265 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart animation_{filename}.mp4"

        os.system(ffmpeg_command)

        for frame_filename in frames:
            os.remove(frame_filename)

    print("Saving final plots...")
    plot_all_actions(Uni.a1_past, Uni.a2_past, filename)
    plot_cumulative_rewards(Uni.cumulative_cost_p1_history, Uni.cumulative_cost_p2_history, filename)
    plot_moving_average_rewards(Uni.cumulative_cost_p1_history, Uni.cumulative_cost_p2_history,
                                n_last=16, filename=filename)

    plt.close(Uni.fig)



    
def main():
    set_seed(1994)

    # Run the experiments defined in the Universe class.
    for n_experiment in [0, 1, 2, 3]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        manual_animation(N_game=500, n_experiment=n_experiment, save=True,
                         filename = f"{timestamp}_experiment{n_experiment}")
        print(f"\nExperiment {n_experiment} completed.")
        
    print("\nSimulation completed.")

if __name__ == "__main__":
    main()
