import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union


def visualize_trajectory_bev(data: Union[Dict[str, List], pd.DataFrame], interval: int = 20) -> None:
    """
    Show trajectory in bird's-eye-view (BEV).
    - Past trajectory shown in blue.
    - Current point shown in red.
    
    Args:
        data: Dictionary or DataFrame containing 'x' and 'y' coordinates
        interval: Animation interval in milliseconds
    """
    if isinstance(data, dict):
        xs = data["x"]
        ys = data["y"]
    else:  # DataFrame
        xs = data["x"].values
        ys = data["y"].values
    
    if len(xs) != len(ys):
        raise ValueError("x and y arrays must have the same length")
    
    if len(xs) == 0:
        print("No data to visualize")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Trajectory Visualization (BEV)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'b-', label="Past Trajectory", linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=8, label="Current Position")
    ax.legend(loc='upper right')

    def init():
        x_min, x_max = np.min(xs) - 5, np.max(xs) + 5
        y_min, y_max = np.min(ys) - 5, np.max(ys) + 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(xs[:frame+1], ys[:frame+1])
        point.set_data([xs[frame]], [ys[frame]])
        return line, point

    ani = FuncAnimation(
        fig, update, frames=len(xs), 
        init_func=init, interval=interval, 
        blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()


def visualize_bbx_bev(data: Union[Dict[str, List], pd.DataFrame], interval: int = 20) -> None:
    """
    Show bounding boxes in BEV.
    - Past trajectory (centers) in blue.
    - Current bounding box in red.
    
    Args:
        data: Dictionary or DataFrame containing 'x', 'y', 'w', and 'l' values
        interval: Animation interval in milliseconds
    """
    if isinstance(data, dict):
        xs = data["x"]
        ys = data["y"]
        ws = data["w"]
        ls = data["l"]
    else:  # DataFrame
        xs = data["x"].values
        ys = data["y"].values
        ws = data["w"].values
        ls = data["l"].values
    
    # Validate data
    if not (len(xs) == len(ys) == len(ws) == len(ls)):
        raise ValueError("x, y, w, l arrays must have the same length")
    
    if len(xs) == 0:
        print("No data to visualize")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Bounding Box Visualization (BEV)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'b-', label="Past Trajectory", linewidth=2)
    rect_patch = patches.Rectangle(
        (0, 0), 0, 0, 
        linewidth=2, edgecolor='r', 
        facecolor='none', label="Current BBox"
    )
    ax.add_patch(rect_patch)
    ax.legend(loc='upper right')

    def init():
        x_min, x_max = np.min(xs) - 5, np.max(xs) + 5
        y_min, y_max = np.min(ys) - 5, np.max(ys) + 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        line.set_data([], [])
        rect_patch.set_xy((0, 0))
        rect_patch.set_width(0)
        rect_patch.set_height(0)
        return line, rect_patch

    def update(frame):
        line.set_data(xs[:frame+1], ys[:frame+1])
        rect_patch.set_xy((xs[frame] - ws[frame]/2, ys[frame] - ls[frame]/2))
        rect_patch.set_width(ws[frame])
        rect_patch.set_height(ls[frame])
        return line, rect_patch

    ani = FuncAnimation(
        fig, update, frames=len(xs), 
        init_func=init, interval=interval, 
        blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()



def visualize_confmap(confmap: np.ndarray):
    """
    Visualize confidence map.
    
    :param confmap: Confidence map array
    :param confmap_cfg: Confidence map configuration
    :param radar_cfg: Radar configuration
    :param frame_label: Optional frame label for annotation
    :param save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confidence Map Visualization', fontsize=16, fontweight='bold')
    
    # Plot individual class maps
    for class_id, class_name in enumerate(confmap_cfg.classes):
        row = class_id // 2
        col = class_id % 2
        
        if class_id < 4:  # Only plot if we have space in 2x2 grid
            im = axes[row, col].imshow(confmap[class_id], cmap='hot', aspect='auto', 
                                      extent=[-radar_cfg.maxAngle, radar_cfg.maxAngle, 
                                              radar_cfg.maxRange, 0])
            axes[row, col].set_title(f'{class_name.capitalize()} Confidence Map')
            axes[row, col].set_xlabel('Azimuth (degrees)')
            axes[row, col].set_ylabel('Range (meters)')
            plt.colorbar(im, ax=axes[row, col])
    
    # Plot merged map (sum of all classes)
    merged_map = np.sum(confmap, axis=0)
    im_merged = axes[1, 1].imshow(merged_map, cmap='viridis', aspect='auto',
                                 extent=[-radar_cfg.maxAngle, radar_cfg.maxAngle, 
                                         radar_cfg.maxRange, 0])
    axes[1, 1].set_title('Merged Confidence Map (All Classes)')
    axes[1, 1].set_xlabel('Azimuth (degrees)')
    axes[1, 1].set_ylabel('Range (meters)')
    plt.colorbar(im_merged, ax=axes[1, 1])
    
    # Add object positions if provided
    if frame_label is not None:
        ranges = frame_label['range']
        azimuths = frame_label['azimuth']
        classes = frame_label['class']
        
        colors = {'pedestrian': 'red', 'cyclist': 'green', 'car': 'blue', 
                 'van': 'orange', 'truck': 'purple'}
        
        for range_val, azimuth_val, class_name in zip(ranges, azimuths, classes):
            if class_name in colors:
                for ax in axes.flat:
                    ax.plot(azimuth_val, range_val, 'o', markersize=8,
                           markerfacecolor=colors[class_name], markeredgecolor='white',
                           markeredgewidth=2, label=f'{class_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence map saved to: {save_path}")
    
    plt.show()
    
    return merged_map
