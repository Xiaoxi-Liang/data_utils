#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_convert_mcap.sh"

# ===== 配置区：直接在这里修改 =====
DISK_ROOT="/home/ss-oss1/data/dataset/external_robotic_data/disk-1"
LOG_DIR="${SCRIPT_DIR}/logs_test4"
OSS_OUTPUT_DIR="oss://ss-oss1/data/user/shawks/DATA/test_debug/agilex-real-world-data/disk-1"

# 留空表示全量 episode，否则只处理前 N 个
MAX_EPISODES="2"

# 并发上限（最多同时跑多少个 task）
MAX_JOBS=30

# 要并行处理的 task 名列表
TASKS=(
  "Arrange_the_number_7"
)
# ===== 配置区结束 =====

mkdir -p "${LOG_DIR}"
chmod +x "${RUNNER}"

running_jobs() {
  jobs -rp | wc -l
}

for task in "${TASKS[@]}"; do
  task_root="${DISK_ROOT}/${task}"
  log_path="${LOG_DIR}/${task}.log"
  REPO_ID="${task}"

  if [[ ! -d "${task_root}" ]]; then
    echo "[warn] task root not found, skip: ${task_root}"
    continue
  fi

  while [[ "$(running_jobs)" -ge "${MAX_JOBS}" ]]; do
    sleep 2
  done

  if [[ -n "${MAX_EPISODES}" ]]; then
    nohup "${RUNNER}" --data-root "${task_root}" --repo-id "${REPO_ID}" --oss-output-dir "${OSS_OUTPUT_DIR}" --max-episodes "${MAX_EPISODES}" > "${log_path}" 2>&1 &
  else
    nohup "${RUNNER}" --data-root "${task_root}" --repo-id "${REPO_ID}" --oss-output-dir "${OSS_OUTPUT_DIR}" > "${log_path}" 2>&1 &
  fi

  echo "[start] task=${task}, repo_id=${REPO_ID}, log=${log_path}"
done

echo "[info] all tasks launched. check logs in ${LOG_DIR}"
