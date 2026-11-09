#!/bin/bash
# KB-E6 đầy đủ ở r = 40 (thử; dừng nếu một fold quá lâu). Cùng mã, cùng output với v2.
cd /home/dev/FKGE_BRSET_Tan_ext
CUDA_VISIBLE_DEVICES=5 /home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python -B -X utf8 -u kb_r.py --device cuda --r 40 \
    >> outputs/kb_r_logs/v2_full_40.log 2>&1
echo "R40_DONE $(date '+%F %T')" >> outputs/kb_r_logs/v2_full_40.log
