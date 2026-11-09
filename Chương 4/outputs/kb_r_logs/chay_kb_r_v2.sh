#!/bin/bash
# KB-E6 với mã đã sửa (attribution lưu sẵn ma trận chuyển vị, trùng từng bit với v3; đo thời gian theo khối vừa cache).
# Chạy lại cùng script để tiếp tục. Đầy đủ R1–R4: r = 8, 16, 23 và r = 32; chỉ R4: mọi r.
cd /home/dev/FKGE_BRSET_Tan_ext
PY=/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python
LOG=/home/dev/FKGE_BRSET_Tan_ext/outputs/kb_r_logs
CUDA_VISIBLE_DEVICES=5 $PY -B -X utf8 -u kb_r.py --device cuda --r 8 16 23 >> $LOG/v2_full_8_16_23.log 2>&1 &
sleep 30
CUDA_VISIBLE_DEVICES=5 $PY -B -X utf8 -u kb_r.py --device cuda --r 32 >> $LOG/v2_full_32.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $PY -B -X utf8 -u kb_r.py --cost-only --device cpu --output outputs/kb_r_cost \
    --reuse-prepared outputs/kb_r_v1_ma_cu >> $LOG/v2_cost.log 2>&1 &
wait
$PY -B -X utf8 kb_r.py --stage report >> $LOG/v2_report.log 2>&1
$PY -B -X utf8 kb_r.py --stage report --output outputs/kb_r_cost >> $LOG/v2_report.log 2>&1
echo "V2_ALL_DONE $(date '+%F %T')" >> $LOG/v2_report.log
