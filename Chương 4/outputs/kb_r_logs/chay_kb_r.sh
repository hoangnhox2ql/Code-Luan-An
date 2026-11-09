#!/bin/bash
# KB-E6 trên BRSET thật: hai tiến trình, mỗi tiến trình một GPU; chạy lại cùng script để tiếp tục (bỏ qua (r, fold) đã xong).
cd /home/dev/FKGE_BRSET_Tan_ext
PY=/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python
LOG=/home/dev/FKGE_BRSET_Tan_ext/outputs/kb_r_logs
CUDA_VISIBLE_DEVICES=4 $PY -B -X utf8 -u kb_r.py --device cuda --r 8 16 23 32 >> $LOG/kb_r_A.log 2>&1 &
sleep 20
CUDA_VISIBLE_DEVICES=5 $PY -B -X utf8 -u kb_r.py --device cuda --r 40 48 >> $LOG/kb_r_B.log 2>&1 &
wait
$PY -B -X utf8 kb_r.py --stage report >> $LOG/kb_r_report.log 2>&1
echo "ALL_DONE $(date '+%F %T')" >> $LOG/kb_r_report.log
