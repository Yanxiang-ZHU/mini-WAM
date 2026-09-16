#!/bin/bash
# Run all remaining trainings sequentially, logging each to logs/.
set -e
cd "/d/ICpro/Wan Lab/4mock"
mkdir -p logs
PY=.venv/Scripts/python.exe

echo "[$(date +%H:%M:%S)] training world model" | tee -a logs/all.log
$PY training/train_world_model.py --config configs/world_model.yaml 2>&1 | grep -vE "nested_tensor|self.encoder" | tee -a logs/world_model.log

echo "[$(date +%H:%M:%S)] training action expert (with subgoal)" | tee -a logs/all.log
$PY training/train_action_expert.py --config configs/action_expert.yaml 2>&1 | grep -vE "nested_tensor|self.encoder" | tee -a logs/action_expert.log

echo "[$(date +%H:%M:%S)] training action chunk policy (no subgoal)" | tee -a logs/all.log
$PY training/train_action_expert.py --config configs/action_chunk_policy.yaml 2>&1 | grep -vE "nested_tensor|self.encoder" | tee -a logs/action_chunk.log

echo "[$(date +%H:%M:%S)] ALL TRAININGS DONE" | tee -a logs/all.log
