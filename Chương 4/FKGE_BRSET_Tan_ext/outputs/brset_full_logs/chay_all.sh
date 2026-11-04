#!/bin/bash
# Lượt chính + bổ sung Chương 4 trên GPU, như MODE="all" của CHAY_CHUONG4_COLAB.ipynb.
# Chạy lại cùng script để tiếp tục (--resume bỏ qua cấu hình/job đã xong).
set -o pipefail
cd /home/dev/FKGE_BRSET_Tan_ext
PY=/home/dev/FKGMM_BRSET3/.venv_tan_ext_gpu/bin/python
OUT=/home/dev/FKGE_BRSET_Tan_ext/outputs/brset_full
LOGS=/home/dev/FKGE_BRSET_Tan_ext/outputs/brset_full_logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
echo "===== $(date '+%F %T') run.py (GPU $CUDA_VISIBLE_DEVICES)" >> $LOGS/run_main.log
$PY -B -X utf8 -u run.py --device cuda --block-size 128 --output $OUT --resume --folds 1 2 3 4 5 >> $LOGS/run_main.log 2>&1 || { echo "RUN_MAIN_FAILED $(date '+%F %T')" >> $LOGS/run_main.log; exit 1; }
RESUME=""; [ -f $OUT/extra/extra_manifest.json ] && RESUME="--resume"
echo "===== $(date '+%F %T') extra.py $RESUME" >> $LOGS/run_extra.log
$PY -B -X utf8 -u extra.py --output $OUT --device cuda --block-size 128 --only kb2,kb3,kb4,kb5,kb6,ablation,baseline --folds 1 2 3 4 5 $RESUME >> $LOGS/run_extra.log 2>&1 || { echo "RUN_EXTRA_FAILED $(date '+%F %T')" >> $LOGS/run_extra.log; exit 1; }
echo "ALL_DONE $(date '+%F %T')" >> $LOGS/run_extra.log
