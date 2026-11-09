#!/bin/bash
# KB-E6, chỉ đo R4 (không giáo viên, không huấn luyện) cho mọi r, 5 fold; dùng lại dữ liệu đã chuẩn bị của outputs/kb_r.
cd /home/dev/FKGE_BRSET_Tan_ext
PY=/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python
LOG=/home/dev/FKGE_BRSET_Tan_ext/outputs/kb_r_logs
export CUDA_VISIBLE_DEVICES=""
for R in "40 48" "8 16 23 32"; do
  $PY -B -X utf8 -u kb_r.py --cost-only --device cpu --output outputs/kb_r_cost --reuse-prepared outputs/kb_r --r $R >> $LOG/kb_r_cost.log 2>&1 || { echo "COST_FAILED $(date '+%F %T')" >> $LOG/kb_r_cost.log; exit 1; }
done
echo "COST_ALL_DONE $(date '+%F %T')" >> $LOG/kb_r_cost.log
