"""
Convert MCAP ROS2 bag files from disk-1 to LeRobot format.

This script converts MCAP ROS2 bag files to LeRobot video-pointer format.
It extracts joint states (200Hz) and images (30Hz) from MCAP files,
downsamples joint data to 30Hz, and outputs in LeRobot format.

Expected source layout:
/home/ss-oss1/data/dataset/external_robotic_data/disk-1/
  <task_name>/
    info.json
    <episode_name>/
      metadata.yaml
      <episode_name>_0.mcap
      <episode_name>_0_info.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[warning] scipy not available. Will use simple downsampling instead of interpolation.")

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
except ImportError:
    lerobot_dataset_module = None

# Try to import rosbag2_py (preferred method)
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState, CompressedImage
    ROSBAG2_AVAILABLE = True
except ImportError:
    ROSBAG2_AVAILABLE = False
    print("[warning] rosbag2_py not available. Will try mcap library.")

# Try to import mcap reader (fallback)
if not ROSBAG2_AVAILABLE:
    try:
        from mcap.reader import make_reader
        try:
            from mcap_ros2.reader import read_ros2_messages
        except ImportError:
            from mcap_ros2_support.reader import read_ros2_messages
        MCAP_AVAILABLE = True
    except ImportError:
        MCAP_AVAILABLE = False
        print("[warning] mcap library not available either.")
else:
    MCAP_AVAILABLE = False

# Constants
CHUNK_SIZE = 1000
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FPS = 30
JOINT_FPS = 200 
JOINT_DOWNSAMPLE_RATIO = JOINT_FPS // TARGET_FPS  # 200 / 30 = 6.67, use 6 or 7


def _motor_names() -> list[str]:
    """
    Return motor names for RobotWin (14 DOF: 7 per arm).
    
    Names are extracted from MCAP JointState messages:
    - Left arm: joint0, joint1, joint2, joint3, joint4, joint5, joint6
    - Right arm: joint0, joint1, joint2, joint3, joint4, joint5, joint6
    Note: joint6 is typically the gripper joint.
    """
    return [
        "left_joint0",
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",  # Typically the gripper
        "right_joint0",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",  # Typically the gripper
    ]


def _video_pointer_struct():
    """Define PyArrow struct for video pointers."""
    return pa.struct([
        ("path", pa.string()),
        ("timestamp", pa.float64()),
        ("frame_index", pa.int64()),
    ])


def configure_lerobot_home(output_dir: Path | None) -> Path:
    """Configure LeRobot home directory."""
    if output_dir is None:
        if lerobot_dataset_module is not None:
            return Path(lerobot_dataset_module.HF_LEROBOT_HOME)
        return Path(os.environ.get("HF_LEROBOT_HOME", "./lerobot_data_disk1")).expanduser().resolve()

    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(resolved)
    if lerobot_dataset_module is not None:
        lerobot_dataset_module.HF_LEROBOT_HOME = resolved
    return resolved


def probe_video_codec_info(video_path: Path) -> dict:
    """Probe video codec info using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"video_codec": "h264", "pix_fmt": "yuv420p", "width": 640, "height": 480, "fps": 30.0}
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return {"video_codec": "h264", "pix_fmt": "yuv420p", "width": 640, "height": 480, "fps": 30.0}
    s = streams[0]
    codec = s.get("codec_name", "h264")
    pix_fmt = s.get("pix_fmt", "yuv420p")
    w = int(s.get("width", 640))
    h = int(s.get("height", 480))
    r = s.get("r_frame_rate", "30/1")
    if "/" in str(r):
        num, den = map(int, str(r).split("/"))
        fps = num / den if den else 30.0
    else:
        fps = float(r) if r else 30.0
    return {"video_codec": codec, "pix_fmt": pix_fmt, "width": w, "height": h, "fps": fps}


def save_video_with_ffmpeg(
    source_path: Path,
    target_path: Path,
    num_frames: int,
    target_width: int,
    target_height: int,
    fps: int = 30,
) -> None:
    """Save video using ffmpeg with specified parameters."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(source_path),
        "-vf", f"scale={target_width}:{target_height}",
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source_path}: {result.stderr}")


def create_dataset_structure_video_pointer(
    dataset_root: Path,
    repo_id: str,
    fps: int,
    image_height: int,
    image_width: int,
    cameras: list[str],
) -> dict:
    """Create video-pointer mode dataset structure."""
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "data").mkdir(parents=True, exist_ok=True)
    (dataset_root / "videos").mkdir(parents=True, exist_ok=True)

    motors = _motor_names()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(motors)],
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(motors)],
            "names": [motors],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    video_info = {
        "video_codec": "h264",
        "fps": float(fps),
        "width": image_width,
        "height": image_height,
        "pix_fmt": "yuv420p",
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, image_height, image_width],
            "names": ["channels", "height", "width"],
            "info": {"video_info": video_info},
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "Cobot Magic",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    info_path = meta_dir / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    return info


def write_episode_parquet_with_pointers(
    dataset_root: Path,
    episode_index: int,
    chunk_name: str,
    frames: list[dict],
    video_rel_paths: dict[str, str],
    fps: int,
    *,
    index_offset: int = 0,
    task_index: int = 0,
) -> Path:
    """Write episode parquet with video pointers."""
    chunk_dir = dataset_root / "data" / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"

    n = len(frames)
    timestamps = np.arange(n, dtype=np.float64) / fps

    obs_state = pa.array([f["observation.state"].tolist() for f in frames], type=pa.list_(pa.float32()))
    action = pa.array([f["action"].tolist() for f in frames], type=pa.list_(pa.float32()))

    def make_pointer_col(video_path: str) -> pa.Array:
        return pa.array(
            [
                {"path": video_path, "timestamp": float(ts), "frame_index": i}
                for i, ts in enumerate(timestamps)
            ],
            type=_video_pointer_struct(),
        )

    columns = {
        "observation.state": obs_state,
        "action": action,
        "timestamp": pa.array(timestamps),
        "frame_index": pa.array(np.arange(n, dtype=np.int64())),
        "episode_index": pa.array([episode_index] * n, type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + n, dtype=np.int64())),
        "task_index": pa.array([task_index] * n, type=pa.int64()),
    }
    
    # Add video pointer columns for all cameras
    # for cam_name, video_path in video_rel_paths.items():
    #     columns[f"observation.images.{cam_name}"] = make_pointer_col(video_path)
    table = pa.table(columns) 
    pq.write_table(table, parquet_path)
    return parquet_path


def _compute_episode_stats_minimal(
    frames: list[dict],
    *,
    episode_index: int,
    index_offset: int,
    task_index: int,
    fps: int = 30,
) -> dict:
    """Compute episode statistics."""
    states = np.array([f["observation.state"] for f in frames], dtype=np.float32)
    actions = np.array([f["action"] for f in frames], dtype=np.float32)
    n = len(frames)
    base = {
        "observation.state": {
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "count": [n],
        },
        "action": {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "count": [n],
        },
    }
    # Note: Video pointer columns don't need image statistics
    # The stats are kept minimal for compatibility with LeRobot dataset format
    ts = np.arange(n, dtype=np.float64) / fps
    base["timestamp"] = {"min": [float(ts.min())], "max": [float(ts.max())], "mean": [float(ts.mean())], "std": [float(ts.std())], "count": [n]}
    base["frame_index"] = {"min": [0], "max": [n - 1], "mean": [(n - 1) / 2], "std": [float(np.sqrt((n * n - 1) / 12))], "count": [n]}
    base["episode_index"] = {"min": [episode_index], "max": [episode_index], "mean": [float(episode_index)], "std": [0.0], "count": [n]}
    idx_std = float(np.sqrt((n * n - 1) / 12)) if n > 1 else 0.0
    base["index"] = {"min": [index_offset], "max": [index_offset + n - 1], "mean": [index_offset + (n - 1) / 2], "std": [idx_std], "count": [n]}
    base["task_index"] = {"min": [task_index], "max": [task_index], "mean": [float(task_index)], "std": [0.0], "count": [n]}
    return base


def update_chunk_episode_json(
    dataset_root: Path,
    episode_parquet_rel_path: Path,
    episode_record: dict,
    chunk_json_cache: dict[str, dict],
) -> tuple[str, Path]:
    """Update chunk episode JSON."""
    chunk_rel_dir = episode_parquet_rel_path.parent
    chunk_key = chunk_rel_dir.as_posix()
    stats_dir = dataset_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    existing = chunk_json_cache.get(chunk_key)
    if existing is None:
        new_payload = {
            "chunk": chunk_rel_dir.name,
            "chunk_path": chunk_key,
            "episodes": [episode_record],
        }
    else:
        # Build a new payload first so failures won't partially mutate cache.
        new_payload = {
            "chunk": existing["chunk"],
            "chunk_path": existing["chunk_path"],
            "episodes": [*existing["episodes"], episode_record],
        }
    json_path = stats_dir / f"{chunk_rel_dir.name}_episodes_state_action.json"
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)
    chunk_json_cache[chunk_key] = new_payload
    return chunk_key, json_path


def write_alignment_stats_summary(all_episode_stats: list[dict], log_path: Path):
    """
    Write aggregated alignment statistics to a log file.
    
    Args:
        all_episode_stats: List of dictionaries, each containing episode alignment stats
        log_path: Path to save the log file
    """
    import json
    from datetime import datetime
    
    if not all_episode_stats:
        return
    
    # Aggregate all time differences across all episodes
    all_diffs_ms = []
    episode_summaries = []
    
    for ep_stat in all_episode_stats:
        ep_diffs_ms = []
        
        # Collect time differences for this episode
        for stat_list in ep_stat["alignment_stats"]["joint_left"]:
            ep_diffs_ms.append(stat_list["diff_ms"])
        for stat_list in ep_stat["alignment_stats"]["joint_right"]:
            ep_diffs_ms.append(stat_list["diff_ms"])
        for cam_topic in ep_stat["alignment_stats"]["cameras"]:
            for stat_list in ep_stat["alignment_stats"]["cameras"][cam_topic]:
                ep_diffs_ms.append(stat_list["diff_ms"])
        
        if ep_diffs_ms:  
            all_diffs_ms.extend(ep_diffs_ms)
            episode_summaries.append({
                "task_name": ep_stat["task_name"],
                "episode_name": ep_stat["episode_name"],
                "mcap_file": ep_stat["mcap_file"],
                "max_alignment_error_ms": float(max(ep_diffs_ms)),
                "mean_alignment_error_ms": float(np.mean(ep_diffs_ms)),
                "median_alignment_error_ms": float(np.median(ep_diffs_ms)),
                "std_alignment_error_ms": float(np.std(ep_diffs_ms)),
                "total_frames": len(ep_stat["alignment_stats"]["joint_left"])
            })
    
    if len(all_diffs_ms) == 0:
        return
    
    # Overall statistics
    overall_max = max(all_diffs_ms)
    overall_mean = np.mean(all_diffs_ms)
    overall_median = np.median(all_diffs_ms)
    overall_std = np.std(all_diffs_ms)
    
    # Prepare output data
    output = {
        "generated_at": datetime.now().isoformat(),
        "overall_summary": {
            "max_alignment_error_ms": float(overall_max),
            "mean_alignment_error_ms": float(overall_mean),
            "median_alignment_error_ms": float(overall_median),
            "std_alignment_error_ms": float(overall_std),
            "total_episodes": len(episode_summaries),
            "total_frames": sum(ep["total_frames"] for ep in episode_summaries)
        },
        "episodes": episode_summaries
    }
    
    # Write to file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[info] Alignment statistics summary saved to: {log_path}")
    print(f"[info] Overall maximum alignment error: {overall_max:.2f} ms")
    print(f"[info] Overall mean alignment error: {overall_mean:.2f} ms")


def detect_camera_topics(mcap_path: Path) -> list[str]:
    """
    Detect all camera image topics in MCAP file.
    
    Returns:
        List of camera topic names (e.g., ['/camera_f/color/image_raw', '/camera_l/color/image_raw'])
    """
    if not ROSBAG2_AVAILABLE:
        return ["/camera_f/color/image_raw"]  # Fallback to default
    
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(mcap_path), storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)
    
    # Get all topics and filter camera image topics
    camera_topics = []
    for topic_metadata in reader.get_all_topics_and_types():
        topic_name = topic_metadata.name
        if '/color/image_raw' in topic_name or '/image_raw' in topic_name:
            camera_topics.append(topic_name)
    
    # Sort for consistent ordering
    camera_topics.sort()
    return camera_topics if camera_topics else ["/camera_f/color/image_raw"]


def extract_data_from_mcap(mcap_path: Path) -> tuple[np.ndarray, dict[str, list[np.ndarray]], float, dict]:
    """
    Extract joint states and images from MCAP file using rosbag2_py API.
    """
    joint_left_msgs = []
    joint_right_msgs = []
    images_by_camera: dict[str, list[dict]] = {}
    start_timestamp = None
    
    # Detect all camera topics
    camera_topics = detect_camera_topics(mcap_path)
    print(f"[info] Detected {len(camera_topics)} camera(s): {[t.split('/')[1] for t in camera_topics]}")
    
    # Initialize image message lists for each camera
    for cam_topic in camera_topics:
        images_by_camera[cam_topic] = []
    
    target_topics = ["/master/joint_left", "/master/joint_right"] + camera_topics
    
    if ROSBAG2_AVAILABLE:
        # Use rosbag2_py API (preferred method)
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(mcap_path),
            storage_id='mcap'
        )
        converter_options = rosbag2_py.ConverterOptions()
        reader.open(storage_options, converter_options)
        
        # Get topic types for deserialization
        topic_types = {}
        for topic_metadata in reader.get_all_topics_and_types():
            topic_types[topic_metadata.name] = topic_metadata.type
        
        # Read all messages
        print(f"[info] Reading MCAP file: {mcap_path.name}")
        msg_count = 0
        
        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next()
            
            # Extract start timestamp from first message
            if start_timestamp is None:
                start_timestamp = timestamp_ns / 1e9
            
            if topic not in target_topics:
                continue
            
            msg_type = topic_types.get(topic)
            if msg_type is None:
                continue
            
            try:
                # Deserialize message based on topic type
                if topic == "/master/joint_left":
                    msg = deserialize_message(data, JointState)
                    positions = list(msg.position) if hasattr(msg, 'position') else []
                    joint_left_msgs.append(
                        {"timestamp": timestamp_ns / 1e9, "position": positions}
                    )
                    msg_count += 1
                    
                elif topic == "/master/joint_right":
                    msg = deserialize_message(data, JointState)
                    positions = list(msg.position) if hasattr(msg, 'position') else []
                    joint_right_msgs.append(
                        {"timestamp": timestamp_ns / 1e9, "position": positions}
                    )
                    msg_count += 1

                elif topic in camera_topics:
                    # Handle any camera topic
                    msg = deserialize_message(data, CompressedImage)
                    img_data = msg.data if hasattr(msg, 'data') else bytes(msg)
                    # Convert array.array to bytes if needed
                    if not isinstance(img_data, bytes):
                        import array
                        if isinstance(img_data, array.array):
                            img_data = img_data.tobytes()
                        else:
                            img_data = bytes(img_data)
                    
                    if isinstance(img_data, bytes):
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images_by_camera[topic].append(
                                {"timestamp": timestamp_ns / 1e9, "data": img_rgb}
                            )
                            msg_count += 1
            except Exception as e:
                print(f"[warning] Failed to deserialize message on {topic} at {timestamp_ns}: {e}")
                continue
        
        # Note: SequentialReader doesn't have a close() method, it's managed automatically
        print(f"[info] Read {msg_count} messages from target topics")
        
    elif MCAP_AVAILABLE:
        # Fallback to mcap library (old method, may have issues with tf2 messages)
        raise RuntimeError("mcap library fallback not implemented. Please use rosbag2_py.")
    else:
        raise RuntimeError(
            "Neither rosbag2_py nor mcap library available. "
            "Please install ROS2 with rosbag2-storage-mcap or pip install mcap mcap-ros2-support"
        )
    
    # Sort by timestamp
    joint_left_msgs.sort(key=lambda x: x["timestamp"])
    joint_right_msgs.sort(key=lambda x: x["timestamp"])
    for cam_topic in camera_topics:
        images_by_camera[cam_topic].sort(key=lambda x: x["timestamp"])
    
    # Check if we have valid data
    if not joint_left_msgs or not joint_right_msgs:
        raise ValueError(f"No valid joint data found in {mcap_path}")
    
    # Check if we have at least one camera with images
    has_images = any(len(images_by_camera[cam]) > 0 for cam in camera_topics)
    if not has_images:
        raise ValueError(f"No valid image data found in {mcap_path}")
    
    # Use the first camera with images as reference for timestamp alignment
    reference_camera = None
    for cam_topic in camera_topics:
        if len(images_by_camera[cam_topic]) > 0:
            reference_camera = cam_topic
            break
    
    if reference_camera is None:
        raise ValueError(f"No camera with images found in {mcap_path}")

    image_msgs = images_by_camera[reference_camera]
    
    # Extract joint positions (7 DOF per arm: 6 joints + 1 gripper)
    left_positions_list = [msg["position"] for msg in joint_left_msgs]
    right_positions_list = [msg["position"] for msg in joint_right_msgs]
    
    # Create arrays with fixed size of 7
    left_joints = np.zeros((len(left_positions_list), 7))
    right_joints = np.zeros((len(right_positions_list), 7))
    
    for i, pos in enumerate(left_positions_list):
        left_joints[i, :len(pos)] = pos
    for i, pos in enumerate(right_positions_list):
        right_joints[i, :len(pos)] = pos
    
    # Get timestamps for interpolation
    reference_timestamps = np.array([msg["timestamp"] for msg in image_msgs])
    joint_left_timestamps = np.array([msg["timestamp"] for msg in joint_left_msgs])
    joint_right_timestamps = np.array([msg["timestamp"] for msg in joint_right_msgs])

    def find_nearest_forward(timestamp, timestamps_array, start_idx=0):
        """
        Find the nearest timestamp forward from start_idx.
        Searches from start_idx onwards and returns the index of the nearest timestamp.
        This avoids backtracking and ensures forward-only matching.
        """
        if start_idx >= len(timestamps_array):
            # Already at or past the end, return last valid index
            return max(0, len(timestamps_array) - 1)
        
        # Search forward from start_idx
        remaining_timestamps = timestamps_array[start_idx:]
        if len(remaining_timestamps) == 0:
            # No remaining timestamps, return last valid index
            return max(0, start_idx - 1)
        
        # Find the nearest timestamp in the remaining array
        time_diffs = np.abs(remaining_timestamps - timestamp)
        nearest_idx_in_remaining = np.argmin(time_diffs)
        return start_idx + nearest_idx_in_remaining
    
    # Align joint states to reference camera timestamps
    # Start from second frame (first frame aligns with second frame)
    if len(reference_timestamps) < 2:
        raise ValueError("Need at least 2 reference frames for alignment")
    
    # Use second frame onwards as reference (first frame aligns with second)
    reference_timestamps_aligned = reference_timestamps[1:]
    
    # Record alignment statistics
    alignment_stats = {
        "joint_left": [],
        "joint_right": [],
        "cameras": {cam: [] for cam in camera_topics}
    }
    
    left_indices = []
    right_indices = []
    left_start_idx = 0
    right_start_idx = 0
    
    for ref_ts in reference_timestamps_aligned:
        # Find nearest left and right joint timestamps forward from current positions
        left_idx = find_nearest_forward(ref_ts, joint_left_timestamps, left_start_idx)
        right_idx = find_nearest_forward(ref_ts, joint_right_timestamps, right_start_idx)
        
        # Get the actual timestamps for left and right joints
        left_ts = joint_left_timestamps[left_idx]
        right_ts = joint_right_timestamps[right_idx]
        
        # Find which joint timestamp is closer to reference timestamp
        left_diff = abs(left_ts - ref_ts)
        right_diff = abs(right_ts - ref_ts)

        # Use the joint timestamp that is closer to reference as the alignment target
        if left_diff <= right_diff:
            # Left joint is closer, align right joint to left joint's timestamp
            aligned_ts = left_ts
            left_indices.append(left_idx)
            left_start_idx = left_idx
            
            # Find right joint index that matches left joint's timestamp
            right_idx_aligned = find_nearest_forward(aligned_ts, joint_right_timestamps, right_start_idx)
            right_ts_aligned = joint_right_timestamps[right_idx_aligned]
            right_indices.append(right_idx_aligned)
            right_start_idx = right_idx_aligned
            
            # Record alignment statistics (in milliseconds)
            alignment_stats["joint_left"].append({
                "ref_ts": ref_ts,
                "aligned_ts": left_ts,
                "diff_ms": left_diff * 1000
            })
            alignment_stats["joint_right"].append({
                "ref_ts": ref_ts,
                "aligned_ts": right_ts_aligned,
                "diff_ms": abs(right_ts_aligned - aligned_ts) * 1000
            })
        else:
            # Right joint is closer, align left joint to right joint's timestamp
            aligned_ts = right_ts
            right_indices.append(right_idx)
            right_start_idx = right_idx
            
            # Find left joint index that matches right joint's timestamp
            left_idx_aligned = find_nearest_forward(aligned_ts, joint_left_timestamps, left_start_idx)
            left_ts_aligned = joint_left_timestamps[left_idx_aligned]
            left_indices.append(left_idx_aligned)
            left_start_idx = left_idx_aligned
            
            # Record alignment statistics (in milliseconds)
            alignment_stats["joint_right"].append({
                "ref_ts": ref_ts,
                "aligned_ts": right_ts,
                "diff_ms": right_diff * 1000
            })
            alignment_stats["joint_left"].append({
                "ref_ts": ref_ts,
                "aligned_ts": left_ts_aligned,
                "diff_ms": abs(left_ts_aligned - aligned_ts) * 1000
            })
    
    # Extract joint data using the aligned indices
    left_joints_interp = left_joints[left_indices]
    right_joints_interp = right_joints[right_indices]
    
    # Combine interpolated joints
    joint_states_interp = np.concatenate([left_joints_interp, right_joints_interp], axis=1).astype(np.float32)
    
    # Align images for all cameras to reference camera timestamps
    images_by_camera_final: dict[str, list[np.ndarray]] = {}
    
    for cam_topic in camera_topics:
        if len(images_by_camera[cam_topic]) == 0:
            continue
        
        cam_timestamps = np.array([msg["timestamp"] for msg in images_by_camera[cam_topic]])
        cam_images = [msg["data"] for msg in images_by_camera[cam_topic]]
        
        # Align camera images to reference timestamps (from second frame onwards)
        cam_images_interp = []
        cam_start_idx = 0
        
        for ref_ts in reference_timestamps_aligned:
            cam_idx = find_nearest_forward(ref_ts, cam_timestamps, cam_start_idx)
            cam_ts = cam_timestamps[cam_idx]
            cam_images_interp.append(cam_images[cam_idx])
            cam_start_idx = cam_idx
            
            # Record alignment statistics (in milliseconds)
            alignment_stats["cameras"][cam_topic].append({
                "ref_ts": ref_ts,
                "aligned_ts": cam_ts,
                "diff_ms": abs(cam_ts - ref_ts) * 1000
            })
        
        images_by_camera_final[cam_topic] = cam_images_interp

    # Ensure same length - if lengths don't match, remove the last frame(s)
    all_lengths = [len(joint_states_interp)] + [len(images_by_camera_final[cam]) for cam in images_by_camera_final]
    min_len = min(all_lengths)
    
    # Trim all data to the same length (remove last frame if lengths don't match)
    joint_states_final = joint_states_interp[:min_len]
    for cam_topic in images_by_camera_final:
        images_by_camera_final[cam_topic] = images_by_camera_final[cam_topic][:min_len]
    
    # Trim alignment stats to match final length
    alignment_stats["joint_left"] = alignment_stats["joint_left"][:min_len]
    alignment_stats["joint_right"] = alignment_stats["joint_right"][:min_len]
    for cam_topic in alignment_stats["cameras"]:
        alignment_stats["cameras"][cam_topic] = alignment_stats["cameras"][cam_topic][:min_len]
    
    return joint_states_final, images_by_camera_final, start_timestamp, alignment_stats


def extract_data_from_mcap_alternative(mcap_path: Path, temp_dir: Path) -> tuple[np.ndarray, dict[str, list[np.ndarray]], float]:
    """
    Alternative method: Use rosbag2 command-line tools to extract data.
    This is a fallback if mcap library is not available.
    
    Returns:
        joint_states: (N, 14) array of joint states
        images_by_camera: dict mapping camera topic to list of (H, W, 3) RGB images
        start_timestamp: start timestamp in seconds
    """
    # This is a placeholder - would need rosbag2 CLI tools
    # For now, raise an error suggesting installation
    raise RuntimeError(
        "MCAP library not available. Please install:\n"
        "  pip install mcap mcap-ros2-support\n"
        "Or use rosbag2 CLI tools to extract data first."
    )


def create_video_from_images(images: list[np.ndarray], output_path: Path, fps: int = 30) -> None:
    """Create video file from list of images."""
    if not images:
        raise ValueError("No images provided")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get image dimensions
    h, w = images[0].shape[:2]
    
    # Use ffmpeg to create video
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save images temporarily
        tmp_images_dir = Path(tmpdir) / "images"
        tmp_images_dir.mkdir()
        
        for i, img in enumerate(images):
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = tmp_images_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(img_path), img_bgr)
        
        # Create video using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-framerate", str(fps),
            "-i", str(tmp_images_dir / "frame_%06d.jpg"),
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def normalize_oss_path(path_str: str) -> str:
    """
    标准化 OSS 路径格式。
    
    Args:
        path_str: 路径字符串（可以是本地路径或 OSS 路径）
    
    Returns:
        标准化后的 OSS 路径（oss://... 格式）
    """
    normalized = path_str.strip()
    if normalized.startswith("/home/"):
        return normalized.replace("/home/", "oss://", 1)
    if normalized.startswith("oss:/") and not normalized.startswith("oss://"):
        return normalized.replace("oss:/", "oss://", 1)
    return normalized


def join_oss_path(base_oss_path: str, relative_path: str | Path) -> str:
    """
    拼接 OSS 路径。
    
    Args:
        base_oss_path: OSS 基础路径
        relative_path: 相对路径
    
    Returns:
        拼接后的完整 OSS 路径
    """
    rel = Path(relative_path).as_posix().lstrip("/")
    return f"{normalize_oss_path(base_oss_path).rstrip('/')}/{rel}"


def run_with_retry(command: str, max_retries: int = 3, delay_seconds: int = 1) -> bool:
    """
    执行命令并在失败时自动重试。
    
    Args:
        command: 要执行的系统命令
        max_retries: 最大重试次数，默认3次
        delay_seconds: 重试间隔秒数，默认1秒
    
    Returns:
        命令是否最终执行成功
    """
    for attempt in range(max_retries):
        exit_code = os.system(command)
        
        if exit_code == 0:
            return True

        if attempt < max_retries - 1:
            print(f"[warn] Command failed (exit code: {exit_code}), retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    
    print(f"[error] Command failed after {max_retries} retries")
    return False


def safe_delete(file_path: Path) -> None:
    """
    安全删除文件。
    
    Args:
        file_path: 要删除的文件路径
    """
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass  # 文件不存在，忽略
    except Exception as exc:
        print(f"[warn] Failed to delete {file_path}: {exc}")


def oss_mv_and_del_file(local_path: Path, oss_target_path: str) -> None:
    """
    上传文件到 OSS 并删除本地文件。
    
    Args:
        local_path: 本地文件路径
        oss_target_path: OSS 目标路径
    """
    normalized_target = normalize_oss_path(oss_target_path)
    command = f'ossutil cp "{local_path}" "{normalized_target}" --update'
    if run_with_retry(command):
        safe_delete(local_path)
    else:
        # Hard-fail: if upload fails we must not leave a partially-built episode/dataset.
        raise RuntimeError(f"Failed to upload {local_path} to {oss_target_path}")


def oss_rm_path(oss_path: str) -> None:
    """Best-effort remove an OSS object (ignore failures)."""
    try:
        command = f'ossutil rm "{normalize_oss_path(oss_path)}" -f'
        os.system(command)
    except Exception as exc:
        print(f"[warn] Failed to rm OSS path {oss_path}: {exc}")


def oss_rm_prefix(oss_prefix: str) -> None:
    """Best-effort recursive remove for OSS prefixes/directories."""
    try:
        command = f'ossutil rm "{normalize_oss_path(oss_prefix)}" -r -f'
        os.system(command)
    except Exception as exc:
        print(f"[warn] Failed to rm OSS prefix {oss_prefix}: {exc}")


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a jsonl file (create parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def upload_dir_files_to_oss_and_delete(local_dir: Path, oss_dataset_root: str, dataset_root: Path) -> None:
    """
    上传目录中的所有文件到 OSS 并删除本地文件。
    
    Args:
        local_dir: 本地目录路径
        oss_dataset_root: OSS 数据集根路径
        dataset_root: 本地数据集根路径（用于计算相对路径）
    """
    if not local_dir.exists():
        return
    
    for local_path in sorted(local_dir.rglob("*")):
        if not local_path.is_file():
            continue
        relative_path = local_path.relative_to(dataset_root)
        oss_target_path = join_oss_path(oss_dataset_root, relative_path)
        oss_mv_and_del_file(local_path, oss_target_path)


def collect_episode_records(data_root: Path, max_episodes: int | None = None) -> list[dict]:
    """
    Collect all episode records from disk-1 directory structure.

    Supports two layouts:
    - data_root points to disk-1 root: /.../disk-1/
      where each task is a subdir containing info.json
    - data_root points to a single task dir: /.../disk-1/<task_name>/
      which directly contains info.json and episode subdirs
    """
    episodes = []

    def _collect_from_task_dir(task_dir: Path) -> None:
        nonlocal episodes

        if not task_dir.is_dir():
            return

        info_json = task_dir / "info.json"
        if not info_json.exists():
            return

        try:
            with open(info_json, "r", encoding="utf-8") as f:
                task_info = json.load(f)
        except Exception:
            return

        task_name = task_dir.name
        task_prompt = task_info.get("task_prompt", task_name)

        for episode_dir in sorted(task_dir.iterdir()):
            if not episode_dir.is_dir() or episode_dir.name == "abnormal":
                continue

            mcap_files = list(episode_dir.glob("*.mcap"))
            if not mcap_files:
                continue

            mcap_path = mcap_files[0]
            episode_name = episode_dir.name

            episodes.append(
                {
                    "task_name": task_name,
                    "task_prompt": task_prompt,
                    "episode_name": episode_name,
                    "episode_dir": episode_dir,
                    "mcap_path": mcap_path,
                }
            )

            if max_episodes is not None and len(episodes) >= max_episodes:
                return

    # If data_root itself is a task dir (has info.json), process only that task.
    if (data_root / "info.json").exists():
        _collect_from_task_dir(data_root)
        return episodes

    # Otherwise, treat as disk-1 root and iterate task subdirs.
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("$"):
            continue
        _collect_from_task_dir(task_dir)
        if max_episodes is not None and len(episodes) >= max_episodes:
            return episodes

    return episodes


def convert_mcap_to_lerobot(
    data_root: Path,
    dataset_root: Path,
    repo_id: str,
    max_episodes: int | None = None,
    oss_output_dir: str | None = None,
) -> tuple[int, int]:
    """Convert MCAP files to LeRobot format."""
    width, height = TARGET_WIDTH, TARGET_HEIGHT
    fps = TARGET_FPS

    # Setup OSS upload if enabled
    oss_dataset_root = None
    if oss_output_dir:
        oss_dataset_root = join_oss_path(oss_output_dir, Path(repo_id))
        print(f"[info] OSS upload enabled. Target: {oss_dataset_root}")

    def rollback_chunk_oss_payload(chunk_name: str) -> list[str]:
        """Delete uploaded data/video payload for one chunk on OSS."""
        if oss_dataset_root is None:
            return []
        deleted_prefixes: list[str] = []
        data_prefix = join_oss_path(oss_dataset_root, Path("data") / chunk_name)
        videos_prefix = join_oss_path(oss_dataset_root, Path("videos") / chunk_name)
        stats_file = join_oss_path(oss_dataset_root, Path("stats") / f"{chunk_name}_episodes_state_action.json")
        for p in (data_prefix, videos_prefix, stats_file):
            oss_rm_prefix(p) if p.endswith(chunk_name) else oss_rm_path(p)
            deleted_prefixes.append(p)
        return deleted_prefixes

    meta_dir = dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    info_path = meta_dir / "info.json"

    # Collect episodes
    episode_records = collect_episode_records(data_root, max_episodes=max_episodes)
    if not episode_records:
        raise RuntimeError("No episode records found")
    
    print(f"[info] Found {len(episode_records)} episodes")
    
    # Detect cameras from first episode
    cameras: list[str] | None = None
    camera_topic_to_name: dict[str, str] = {}
    
    task_to_index: dict[str, int] = {}
    task_order: list[str] = []
    episodes_lines: list[dict] = []
    stats_lines: list[dict] = []
    
    converted = 0
    skipped = 0
    total_frames = 0
    chunk_json_cache: dict[str, dict] = {}
    probed_video_info: dict | None = None
    all_episode_alignment_stats: list[dict] = []
    # Rollback deletion audit (OSS). We will also write per-event jsonl immediately.
    rollback_events: list[dict] = []
    rollback_log_path = meta_dir / "rollback_deletions.jsonl"
    failed_episodes: list[dict] = []
    failed_episodes_log_path = meta_dir / "failed_episodes.jsonl"
    
    # Track chunk changes for OSS upload
    last_chunk_key: str | None = None
    last_chunk_json_path: Path | None = None
    
    for ep_idx, ep_record in tqdm(enumerate(episode_records), total=len(episode_records)):
        mcap_path = ep_record["mcap_path"]
        task_name = ep_record["task_name"]
        task_prompt = ep_record["task_prompt"]
        episode_name = ep_record["episode_name"]

        try:
            # Track uploaded OSS objects for this episode so we can roll back on failure.
            uploaded_oss_paths: list[str] = []
            created_local_paths: list[Path] = []

            # Extract data from MCAP
            if ROSBAG2_AVAILABLE:
                joint_states, images_by_camera, start_ts, alignment_stats = extract_data_from_mcap(mcap_path)
            elif MCAP_AVAILABLE:
                # Fallback: convert dict to single camera format for compatibility
                joint_states, images_by_camera, start_ts, alignment_stats = extract_data_from_mcap(mcap_path)
            else:
                joint_states, images_by_camera, start_ts = extract_data_from_mcap_alternative(mcap_path, dataset_root / "temp")
                alignment_stats = {"joint_left": [], "joint_right": [], "cameras": {}}
            # Defer recording alignment stats until episode is fully successful.
            episode_alignment_record = {
                "task_name": task_name,
                "episode_name": episode_name,
                "mcap_file": str(mcap_path),
                "alignment_stats": alignment_stats,
            }

            # Detect cameras on first episode
            if cameras is None:
                camera_topics = list(images_by_camera.keys())
                # Map camera topic to camera name (e.g., /camera_f/color/image_raw -> cam_f)
                camera_topic_to_name = {}
                cameras = []
                for cam_topic in sorted(camera_topics):
                    # Extract camera name from topic (e.g., /camera_f/color/image_raw -> cam_f)
                    cam_name = cam_topic.split('/')[1].replace('camera_', 'cam_')
                    camera_topic_to_name[cam_topic] = cam_name
                    cameras.append(cam_name)
                
                print(f"[info] Detected {len(cameras)} camera(s): {cameras}")
                
                # Create dataset structure with detected cameras
                create_dataset_structure_video_pointer(
                    dataset_root=dataset_root,
                    repo_id=repo_id,
                    fps=fps,
                    image_height=height,
                    image_width=width,
                    cameras=cameras,
                )









            # Alignment / frame construction will happen below
            task = task_prompt 
            # 
            episode_index = converted
            chunk_idx = episode_index // CHUNK_SIZE
            chunk_name = f"chunk-{chunk_idx:03d}"
            current_chunk_key = (Path("data") / chunk_name).as_posix()
            
            # Create videos for all cameras (and upload)
            video_rel_paths: dict[str, str] = {}
            for cam_topic, cam_name in camera_topic_to_name.items():
                cam_images = images_by_camera.get(cam_topic, [])
     
                
                video_rel_path = f"videos/{chunk_name}/observation.images.{cam_name}/episode_{episode_index:06d}.mp4"
                video_target = dataset_root / video_rel_path
                create_video_from_images(cam_images, video_target, fps=fps)
                created_local_paths.append(video_target)
                video_rel_paths[cam_name] = video_rel_path

                if probed_video_info is None and video_target.exists():
                    probed_video_info = probe_video_codec_info(video_target)

                # Upload video immediately to OSS if enabled, then delete local file
                if oss_dataset_root is not None and video_target.exists():
                    oss_target_path = join_oss_path(oss_dataset_root, video_rel_path)
                    oss_mv_and_del_file(video_target, oss_target_path)
                    uploaded_oss_paths.append(oss_target_path)

            # Use a provisional task index; commit to task map only after episode succeeds.
            is_new_task = task not in task_to_index
            if is_new_task:
                task_idx = len(task_to_index)
            else:
                task_idx = task_to_index[task]
            
            # Create frames (state-action pairs)
            frames = [
                {"observation.state": joint_states[i], "action": joint_states[i + 1]}
                for i in range(len(joint_states) - 1)
            ]

            # Write parquet for this episode (and upload)
            episode_parquet_path = write_episode_parquet_with_pointers(
                dataset_root=dataset_root,
                episode_index=episode_index,
                chunk_name=chunk_name,
                frames=frames,
                video_rel_paths=video_rel_paths,
                fps=fps,
                index_offset=total_frames,
                task_index=task_idx,
            )
            created_local_paths.append(episode_parquet_path)
            
            # Upload parquet file to OSS if enabled
            if oss_dataset_root is not None:
                episode_parquet_rel_path = Path("data") / chunk_name / f"episode_{episode_index:06d}.parquet"
                oss_target_path = join_oss_path(oss_dataset_root, episode_parquet_rel_path)
                oss_mv_and_del_file(episode_parquet_path, oss_target_path)
                uploaded_oss_paths.append(oss_target_path)
            
            num_frames = len(frames)
            stats_minimal = _compute_episode_stats_minimal(
                frames,
                episode_index=episode_index,
                index_offset=total_frames,
                task_index=task_idx,
                fps=fps,
            )

            # Upload previous chunk's JSON first (if we are switching chunks).
            # Do this before current episode metadata commit to keep per-episode commit atomic.
            if (
                oss_dataset_root is not None
                and last_chunk_key is not None
                and last_chunk_key != current_chunk_key
                and last_chunk_json_path is not None
                and last_chunk_json_path.exists()
            ):
                json_rel_path = last_chunk_json_path.relative_to(dataset_root)
                oss_target_path = join_oss_path(oss_dataset_root, json_rel_path)
                oss_mv_and_del_file(last_chunk_json_path, oss_target_path)
                uploaded_oss_paths.append(oss_target_path)

            # ---- Commit episode metadata atomically after all failure-prone steps above ----
            episode_parquet_rel_path = Path("data") / chunk_name / f"episode_{episode_index:06d}.parquet"
            current_chunk_key, current_chunk_json_path = update_chunk_episode_json(
                dataset_root=dataset_root,
                episode_parquet_rel_path=episode_parquet_rel_path,
                episode_record={
                    "episode_chunk": chunk_idx,
                    "episode_index": episode_index,
                    "source_episode_index": ep_idx,
                    "episode_name": episode_name,
                    "task": task,
                    "mcap_path": str(mcap_path),
                    "num_frames": num_frames,
                    "observation.state": joint_states[:-1].tolist(),
                    "action": joint_states[1:].tolist(),
                },
                chunk_json_cache=chunk_json_cache,
            )
            if is_new_task:
                task_to_index[task] = task_idx
                task_order.append(task)
            episodes_lines.append(
                {
                    "episode_chunk": chunk_idx,
                    "episode_index": episode_index,
                    "tasks": [json.dumps({"task": task}, ensure_ascii=False)],
                    "length": num_frames,
                }
            )
            stats_lines.append({"episode_chunk": chunk_idx, "episode_index": episode_index, "stats": stats_minimal})
            all_episode_alignment_stats.append(episode_alignment_record)
            last_chunk_key = current_chunk_key
            last_chunk_json_path = current_chunk_json_path
            
            total_frames += num_frames
            converted += 1
            
            if converted % 20 == 0:
                print(f"[progress] converted={converted}, skipped={skipped}, last_episode={episode_name}")
        
        except Exception as exc:
            # Roll back partial uploads for this episode to avoid broken datasets (missing mp4/parquet).
            if oss_dataset_root is not None:
                deleted: list[str] = []
                for p in uploaded_oss_paths:
                    oss_rm_path(p)
                    deleted.append(p)
                evt = {
                    "episode_name": episode_name,
                    "source_episode_index": ep_idx,
                    "mcap_path": str(mcap_path),
                    "reason_type": type(exc).__name__,
                    "reason": str(exc),
                    "deleted_oss_paths": deleted,
                }
                rollback_events.append(evt)
                append_jsonl(rollback_log_path, evt)
            # Remove local artifacts created by this failed episode in local-output mode.
            for local_path in created_local_paths:
                safe_delete(local_path)
            fail_record = {
                "episode_name": episode_name,
                "source_episode_index": ep_idx,
                "mcap_path": str(mcap_path),
                "reason_type": type(exc).__name__,
                "reason": str(exc),
            }
            failed_episodes.append(fail_record)
            append_jsonl(failed_episodes_log_path, fail_record)
            skipped += 1
            print(f"[warn] skip episode={episode_name}, reason={type(exc).__name__}: {exc}")
    
    # Ensure meta directory exists before writing files
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure meta directory exists before writing files
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metadata files
    tasks_lines = [
        {"task_index": i, "task": json.dumps({"task": t}, ensure_ascii=False)}
        for i, t in enumerate(task_order)
    ]
    with open(episodes_path, "w", encoding="utf-8") as f:
        for line in episodes_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(tasks_path, "w", encoding="utf-8") as f:
        for line in tasks_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    # Update info.json (create if doesn't exist)
    total_chunks = (converted + CHUNK_SIZE - 1) // CHUNK_SIZE if converted > 0 else 0
    total_videos = converted
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        # Create default info if no episodes were converted
        default_cameras = cameras if cameras else ["cam_f"]
        info = create_dataset_structure_video_pointer(
            dataset_root=dataset_root,
            repo_id=repo_id,
            fps=fps,
            image_height=height,
            image_width=width,
            cameras=default_cameras,
        )
    info["total_episodes"] = converted
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_to_index)
    info["total_videos"] = total_videos
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{converted}"}
    if probed_video_info is not None:
        video_info = {
            "video_codec": probed_video_info["video_codec"],
            "fps": float(probed_video_info["fps"]),
            "width": probed_video_info["width"],
            "height": probed_video_info["height"],
            "pix_fmt": probed_video_info["pix_fmt"],
        }
        for cam in cameras:
            key = f"observation.images.{cam}"
            if key in info.get("features", {}):
                info["features"][key]["info"] = {"video_info": video_info}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    
    # Write alignment statistics summary to meta folder
    alignment_stats_path = meta_dir / "alignment_stats.json"
    write_alignment_stats_summary(all_episode_alignment_stats, alignment_stats_path)

    # Write rollback summary (OSS deletions) if any
    if rollback_events:
        summary_path = meta_dir / "rollback_deletions_summary.json"
        summary = {
            "total_events": len(rollback_events),
            "total_deleted_objects": int(sum(len(e.get("deleted_oss_paths", [])) for e in rollback_events)),
            "events": rollback_events,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[info] Rollback deletion log saved to: {rollback_log_path}")
        print(f"[info] Rollback deletion summary saved to: {summary_path}")

    if failed_episodes:
        failed_summary_path = meta_dir / "failed_episodes_summary.json"
        failed_summary = {
            "total_failed_episodes": len(failed_episodes),
            "failed_episodes": failed_episodes,
        }
        with open(failed_summary_path, "w", encoding="utf-8") as f:
            json.dump(failed_summary, f, ensure_ascii=False, indent=2)
        print(f"[info] Failed episodes log saved to: {failed_episodes_log_path}")
        print(f"[info] Failed episodes summary saved to: {failed_summary_path}")
    
    # Upload remaining chunk JSON file if OSS is enabled
    if oss_dataset_root is not None:
        upload_failures: list[dict] = []
        had_chunk_consistency_failure = False
        if last_chunk_json_path is not None and last_chunk_json_path.exists():
            try:
                json_rel_path = last_chunk_json_path.relative_to(dataset_root)
                oss_target_path = join_oss_path(oss_dataset_root, json_rel_path)
                oss_mv_and_del_file(last_chunk_json_path, oss_target_path)
            except Exception as exc:
                chunk_name = last_chunk_json_path.name.replace("_episodes_state_action.json", "")
                deleted_prefixes = rollback_chunk_oss_payload(chunk_name)
                had_chunk_consistency_failure = True
                upload_failures.append(
                    {
                        "stage": "upload_last_chunk_json",
                        "local_path": str(last_chunk_json_path),
                        "reason_type": type(exc).__name__,
                        "reason": str(exc),
                        "rolled_back_prefixes": deleted_prefixes,
                    }
                )
                print(f"[warn] failed upload of last chunk json: {exc}")
        
        # Upload all remaining directories (meta, data, stats, videos)
        print(f"[info] Uploading remaining files to OSS...")
        for subdir in (dataset_root / "data", dataset_root / "stats"):
            try:
                upload_dir_files_to_oss_and_delete(subdir, oss_dataset_root, dataset_root)
            except Exception as exc:
                failure_item = {
                    "stage": "upload_dir_files_to_oss_and_delete",
                    "local_path": str(subdir),
                    "reason_type": type(exc).__name__,
                    "reason": str(exc),
                }
                # If stats upload fails for a chunk json, roll back corresponding chunk payload.
                if subdir.name == "stats":
                    m = re.search(r"(chunk-\d+)_episodes_state_action\.json", str(exc))
                    if m:
                        failed_chunk = m.group(1)
                        deleted_prefixes = rollback_chunk_oss_payload(failed_chunk)
                        failure_item["rolled_back_prefixes"] = deleted_prefixes
                        had_chunk_consistency_failure = True
                upload_failures.append(
                    failure_item
                )
                print(f"[warn] failed upload of directory {subdir}: {exc}")
        # Videos are uploaded per-episode immediately after creation above.

        # Upload meta only when chunk-level consistency is preserved.
        if not had_chunk_consistency_failure:
            try:
                upload_dir_files_to_oss_and_delete(meta_dir, oss_dataset_root, dataset_root)
            except Exception as exc:
                upload_failures.append(
                    {
                        "stage": "upload_dir_files_to_oss_and_delete",
                        "local_path": str(meta_dir),
                        "reason_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                )
                print(f"[warn] failed upload of directory {meta_dir}: {exc}")
        else:
            print("[warn] skipped meta upload due to chunk consistency failure")

        if upload_failures:
            upload_failed_log_path = meta_dir / "upload_failures_summary.json"
            with open(upload_failed_log_path, "w", encoding="utf-8") as f:
                json.dump({"failures": upload_failures}, f, ensure_ascii=False, indent=2)
            print(f"[warn] Some final uploads failed")
            print(f"[warn] Upload failure summary saved to: {upload_failed_log_path}")
        # User requirement: always remove local dataset after finishing OSS flow.
        shutil.rmtree(dataset_root, ignore_errors=True)
        print(f"[info] Finalized OSS flow for {oss_dataset_root} and removed local directory")
    
    print(f"[done] converted={converted}, skipped={skipped}, total_requested={len(episode_records)}")
    
    return converted, skipped


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert MCAP ROS2 bag files to LeRobot format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ss-oss1/data/dataset/external_robotic_data/disk-1"),
        help="Root directory containing MCAP files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="external_robotic_data_disk1",
        help="LeRobot repo id.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./lerobot_data_disk1",
        help="Output directory for LeRobot dataset.",
    )
    parser.add_argument(
        "--oss-output-dir",
        type=str,
        default=None,
        help="Optional OSS directory for uploading.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    lerobot_home = configure_lerobot_home(args.output_dir)
    print(f"[info] lerobot_home={lerobot_home}")
    dataset_root = lerobot_home / args.repo_id
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    convert_mcap_to_lerobot(
        data_root=args.data_root,
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        oss_output_dir=args.oss_output_dir,
    )


if __name__ == "__main__":
    """
    Example usage:
    python convert_mcap_to_lerobot.py \
        --data-root /home/ss-oss1/data/dataset/external_robotic_data/disk-1 \
        --repo-id external_robotic_data_disk1 \
        --max-episodes 100 \
        --output-dir ./lerobot_data_disk1
    """
    main()