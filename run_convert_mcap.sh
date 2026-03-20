#!/bin/bash
# Wrapper script to run convert_mcap_to_lerobot.py in ROS2 environment

set -e

# Source conda and activate ROS2 environment
source /home/xiaoxi/miniconda3/etc/profile.d/conda.sh
conda activate rosbag_humble

# Source ROS2 setup script to set AMENT_PREFIX_PATH
source /home/xiaoxi/miniconda3/envs/rosbag_humble/setup.bash

# Set proxy if needed
export http_proxy=http://localhost:7890
export https_proxy=http://localhost:7890
export HTTP_PROXY=http://localhost:7890
export HTTPS_PROXY=http://localhost:7890
export no_proxy=localhost,127.0.0.1

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python script with all arguments
python "${SCRIPT_DIR}/convert_mcap_to_lerobot.py" "$@"
