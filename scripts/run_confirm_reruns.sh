#!/usr/bin/env bash
# Confirm-reruns for P0P1 outliers (4 jobs, single GPU serial). Launch detached:
#   setsid nohup bash scripts/run_confirm_reruns.sh >> logs/confirm_reruns.log 2>&1 < /dev/null &
# Survives SSH/Cursor disconnect. Skip a step iff results/runs/<run_name>/final_metrics.json exists.
# OOM: retry once (PYTORCH_CUDA_ALLOC_CONF already expandable_segments). Record per-step exitcode.
set -u
trap '' HUP

ROOT="/root/autodl-tmp/rp_lora_v0"
PYTHON="/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python"
LOGDIR="${ROOT}/logs"
RESULTS="${ROOT}/results"
LOCK="/tmp/rp_lora_v0_gpu.lock"
STEPS_LOG="${LOGDIR}/confirm_reruns.steps.tsv"
cd "${ROOT}" || exit 1
mkdir -p "${LOGDIR}"

unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export PYTHONUNBUFFERED=1

ts() { date -Iseconds; }

echo "===== CONFIRM RERUN START $(ts) pid=$$ ppid=${PPID} sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ') ====="
echo "ROOT=${ROOT}"
echo "=== nvidia-smi ==="
timeout 15 nvidia-smi || echo "nvidia-smi timeout/fail"
echo "=== compute apps ==="
timeout 12 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

if [[ ! -f "${ROOT}/results/runs/v0_sft_instrdialog/final_adapter/adapter_model.safetensors" ]]; then
  echo "FAIL missing SFT adapter ${ROOT}/results/runs/v0_sft_instrdialog/final_adapter/adapter_model.safetensors"
  exit 2
fi

exec 9>"${LOCK}"
while ! flock -n 9; do
  echo "$(ts) LOCK_BUSY ${LOCK}; sleep 30 (will take over after previous orchestrator exits)"
  sleep 30
done
echo "$(ts) acquired ${LOCK}"

gpu_busy() {
  if ps -eo pid,cmd | grep -F ' -m core.train' | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  local out
  out="$(timeout 12 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  echo "${out}" | grep -q '[0-9]'
}

wait_gpu_idle() {
  local n=0
  while gpu_busy; do
    n=$((n + 1))
    echo "$(ts) GPU busy; sleep 30 [${n}]"
    timeout 12 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    sleep 30
  done
}

record_step() {
  local name="$1" status="$2" extra="${3:-}"
  echo -e "$(ts)\t${name}\t${status}\t${extra}" >> "${STEPS_LOG}"
  echo "$(ts) STEP ${name} ${status} ${extra}"
}

copy_metrics() {
  local run_name="$1"
  local dest_dir="$2"
  local dest_stem="$3"
  mkdir -p "${dest_dir}"
  local src_json="${RESULTS}/runs/${run_name}/final_metrics.json"
  local src_csv="${RESULTS}/tables/${run_name}_segment_metrics.csv"
  if [[ -f "${src_json}" ]]; then
    cp -f "${src_json}" "${dest_dir}/final_metrics_${dest_stem}.json"
  fi
  if [[ -f "${src_csv}" ]]; then
    cp -f "${src_csv}" "${dest_dir}/segment_metrics_${dest_stem}.csv"
  fi
}

run_one() {
  local name="$1"
  local cfg="$2"
  local exp="$3"
  local run_name="$4"
  local dest_dir="$5"
  local dest_stem="$6"

  local fm="${RESULTS}/runs/${run_name}/final_metrics.json"
  if [[ -f "${fm}" ]]; then
    echo "0" > "${LOGDIR}/${name}.exitcode"
    copy_metrics "${run_name}" "${dest_dir}" "${dest_stem}"
    record_step "${name}" "SKIP" "already_ok run=${RESULTS}/runs/${run_name}"
    return 0
  fi

  if [[ -d "${RESULTS}/runs/${run_name}" ]]; then
    echo "$(ts) incomplete run dir ${RESULTS}/runs/${run_name}; removing before retry"
    rm -rf "${RESULTS}/runs/${run_name}"
  fi

  wait_gpu_idle
  if gpu_busy; then
    echo "FAIL GPU still busy before ${name}"
    echo "98" > "${LOGDIR}/${name}.exitcode"
    record_step "${name}" "FAIL" "gpu_busy"
    return 98
  fi

  echo "===== START ${name} config=${cfg} run_name=${run_name} ====="
  date -Iseconds > "${LOGDIR}/${name}.start"
  "${PYTHON}" -m core.train --config "${cfg}" > "${LOGDIR}/${name}.log" 2>&1
  local ec=$?
  echo "${ec}" > "${LOGDIR}/${name}.exitcode"
  date -Iseconds > "${LOGDIR}/${name}.end"

  if [[ "${ec}" -ne 0 ]]; then
    local logt
    logt="$(tail -c 2500 "${LOGDIR}/${name}.log" 2>/dev/null || true)"
    if echo "${logt}" | grep -qE 'OutOfMemoryError|CUDA out of memory'; then
      echo "$(ts) OOM on ${name}; retry once with expandable_segments"
      wait_gpu_idle
      if [[ -d "${RESULTS}/runs/${run_name}" ]]; then
        rm -rf "${RESULTS}/runs/${run_name}"
      fi
      date -Iseconds > "${LOGDIR}/${name}.start"
      "${PYTHON}" -m core.train --config "${cfg}" > "${LOGDIR}/${name}.log" 2>&1
      ec=$?
      echo "${ec}" > "${LOGDIR}/${name}.exitcode"
      date -Iseconds > "${LOGDIR}/${name}.end"
    fi
  fi

  if [[ "${ec}" -ne 0 ]]; then
    record_step "${name}" "FAIL" "train_exit=${ec}"
    echo "FAIL train ${name} exit=${ec}"
    tail -n 80 "${LOGDIR}/${name}.log" || true
    return "${ec}"
  fi

  "${PYTHON}" "${ROOT}/scripts/check_run_ok.py" "${name}" "${exp}"
  local cec=$?
  if [[ "${cec}" -ne 0 ]]; then
    record_step "${name}" "FAIL" "check_exit=${cec}"
    echo "FAIL check ${name}"
    return "${cec}"
  fi
  if [[ ! -f "${fm}" ]]; then
    record_step "${name}" "FAIL" "missing ${fm}"
    echo "FAIL missing ${fm}"
    return 1
  fi
  copy_metrics "${run_name}" "${dest_dir}" "${dest_stem}"
  record_step "${name}" "PASS" "exit=0 run=${RESULTS}/runs/${run_name}"
  echo "===== PASS ${name} ====="
  return 0
}

: > "${STEPS_LOG}"
echo -e "time\tname\tstatus\textra" >> "${STEPS_LOG}"

TE_DIR="${ROOT}/experiments_v1/v1_teacher_enhance_OPSD/results"
SG_DIR="${ROOT}/experiments_v1/v1_seg_OPSD/results"

# name|cfg|experiment_name|run_name|dest_dir|dest_stem
STEPS=(
  "v1_teacher_enhance_opsd_instrdialog_seed123_rerun2|configs/v1_teacher_enhance_opsd_instrdialog_seed123_rerun2.yaml|v1_teacher_enhance_opsd_instrdialog|v1_teacher_enhance_opsd_instrdialog_seed123_rerun2|${TE_DIR}|instrdialog_seed123_rerun2"
  "v1_teacher_enhance_opsd_instrdialog_seed789|configs/v1_teacher_enhance_opsd_instrdialog_seed789.yaml|v1_teacher_enhance_opsd_instrdialog|v1_teacher_enhance_opsd_instrdialog_seed789|${TE_DIR}|instrdialog_seed789"
  "v1_seg_opsd_k25_instrdialog_seed456_rerun|configs/v1_seg_opsd_k25_instrdialog_seed456_rerun.yaml|v1_seg_opsd_instrdialog_k25|v1_seg_opsd_k25_instrdialog_seed456_rerun|${SG_DIR}|instrdialog_k25_seed456_rerun"
  "v1_seg_opsd_k25_instrdialog_seed789|configs/v1_seg_opsd_k25_instrdialog_seed789.yaml|v1_seg_opsd_instrdialog_k25|v1_seg_opsd_k25_instrdialog_seed789|${SG_DIR}|instrdialog_k25_seed789"
)

echo "=== skip/run plan (preflight, 4 steps) ==="
for spec in "${STEPS[@]}"; do
  IFS='|' read -r name cfg exp run_name dest_dir dest_stem <<<"${spec}"
  if [[ -f "${RESULTS}/runs/${run_name}/final_metrics.json" ]]; then
    echo "SKIP  ${name}"
  else
    echo "RUN   ${name}  ${cfg}"
  fi
done

fail_exit() {
  local ec="$1"
  echo "$(ts) PIPELINE_FAIL exit=${ec}"
  exit "${ec}"
}

for spec in "${STEPS[@]}"; do
  IFS='|' read -r name cfg exp run_name dest_dir dest_stem <<<"${spec}"
  run_one "${name}" "${cfg}" "${exp}" "${run_name}" "${dest_dir}" "${dest_stem}" || fail_exit $?
done

echo "$(ts) ALL_PIPELINE_STEPS_COMPLETED n=4"
echo "===== PIPELINE DONE $(ts) pid=$$ ====="
exit 0
