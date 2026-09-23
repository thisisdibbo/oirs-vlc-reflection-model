#!/bin/sh
# Day 15 freeze: regenerate every figure at the frozen settings.
set -e
cd /home/claude/lifi
: > freeze_run.log
for s in compare_models.py validate_optimizers.py sweep_panel.py sweep_power.py \
         day4_orientation.py day5_mobility.py day6_blockage.py \
         day7_noise.py day8_proposed.py day9_ber.py ; do
  echo "" >> freeze_run.log
  echo "################################################################" >> freeze_run.log
  echo "### $s   started $(date -u +%H:%M:%S)" >> freeze_run.log
  echo "################################################################" >> freeze_run.log
  t0=$(date +%s)
  python3 "$s" >> freeze_run.log 2>&1
  t1=$(date +%s)
  echo "### $s finished in $((t1-t0)) s" >> freeze_run.log
  echo "DONE $s $((t1-t0))s"
done
echo "ALL_FREEZE_RUNS_COMPLETE" >> freeze_run.log
