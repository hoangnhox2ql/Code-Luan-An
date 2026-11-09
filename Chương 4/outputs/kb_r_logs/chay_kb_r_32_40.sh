#!/bin/bash
# Một tiến trình cho r = 32 (tiếp từ fold chưa xong) rồi r = 40: kb_r.py dọn thư mục cache gốc khi tiến trình
# xong mọi (r, fold) của nó, nên các lượt đầy đủ dùng chung output không được chạy song song.
cd /home/dev/FKGE_BRSET_Tan_ext
CUDA_VISIBLE_DEVICES=5 /home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python -B -X utf8 -u kb_r.py --device cuda --r 32 40 \
    >> outputs/kb_r_logs/v2_full_32_40.log 2>&1
/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python -B -X utf8 kb_r.py --stage report >> outputs/kb_r_logs/v2_report.log 2>&1
echo "R32_40_DONE $(date '+%F %T')" >> outputs/kb_r_logs/v2_full_32_40.log
