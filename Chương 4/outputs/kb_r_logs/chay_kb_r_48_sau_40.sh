#!/bin/bash
# Chờ lượt r = 32 → 40 kết thúc (dọn cache xong), rồi chạy KB-E6 đầy đủ ở r = 48 trong một tiến trình riêng.
cd /home/dev/FKGE_BRSET_Tan_ext
until grep -q "R32_40_DONE" outputs/kb_r_logs/v2_full_32_40.log 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=5 /home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python -B -X utf8 -u kb_r.py --device cuda --r 48 \
    >> outputs/kb_r_logs/v2_full_48.log 2>&1
/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python -B -X utf8 kb_r.py --stage report >> outputs/kb_r_logs/v2_report.log 2>&1
echo "R48_DONE $(date '+%F %T')" >> outputs/kb_r_logs/v2_full_48.log
