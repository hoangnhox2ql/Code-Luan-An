#!/bin/bash
# Dừng tiến trình B (r = 40, 48 đầy đủ) ngay sau khi r = 40 fold 1 xong: giáo viên FISA ở r >= 40 mất hàng giờ mỗi fold.
F=/home/dev/FKGE_BRSET_Tan_ext/outputs/kb_r/r40/fold_01/result.json
until [ -f $F ]; do sleep 30; done
sleep 5
for P in $(ps -eo pid,cmd | grep "kb_r.py --device cuda --r 40 48" | grep -v grep | awk '{print $1}'); do kill $P; echo "killed B pid $P $(date '+%F %T')"; done
