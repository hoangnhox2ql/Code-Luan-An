# FKGMM_BRSET_Chuong3 — Chương 3 (FKG-UM / FKG-MM) trên BRSET 16.266 ảnh

Bộ mã độc lập, chỉ cần CPU, không dùng PyTorch. Dữ liệu: `data/fusion_features.csv` (16.266 ảnh, 8.524 bệnh nhân,
10 đặc trưng ảnh GLCM + 13 cột lâm sàng, giữ nguyên giá trị thiếu) và 5 outer fold theo bệnh nhân
`data/fold_assignments.csv` (StratifiedGroupKFold, seed 42; cùng fold với Chương 4).

## Chạy

Colab: tải `FKGMM_BRSET_Chuong3.zip` lên `MyDrive/` rồi mở `CHAY_CHUONG3_COLAB.ipynb` (Runtime CPU là đủ).

Máy riêng (Python 3.10+):
```bash
python -m pip install -r requirements.txt
python -B -X utf8 -m unittest tests.test_ch3 -v                                   # 18 test, khoảng 5 phút
python -B -X utf8 -u chuong3.py all --out outputs/chuong3                          # dữ liệu 7 nhóm + M1–M6, EPS
python -B -X utf8 -u chuong3.py run --out outputs/chuong3_dedup --data outputs/chuong3/data --rule-base dedup_conf
python -B -X utf8 -u chuong3.py audit --out outputs/chuong3                        # tuỳ chọn: FISA-GM vs GM-w
python -B -X utf8 -u chuong3.py all --smoke --out outputs/smoke                    # kiểm tra nhanh, KHÔNG phải BRSET
```
Mọi lệnh chạy tiếp được sau khi bị ngắt với `--resume`; đổi mã/cấu hình/dữ liệu trên cùng `--out` bị từ chối.

## Quy trình

1. **Dữ liệu 7 nhóm** (`chuong3.py prepare`, hàm trong `fusion_strategies.py` trích **nguyên văn** từ
   `Preprocess_kfold_feature_selection.py`, bản gốc ở `reference/preprocess_source/`). Mọi bước điền thiếu, chuẩn hoá,
   chọn đặc trưng, chiếu, SVD và BorderlineSMOTE (seed 42 + fold) chỉ fit trên ảnh huấn luyện của fold.

   | Nhóm | Cách dựng (`config_ch3.json`) |
   |---|---|
   | table | MinMax, MI chọn 13 cột lâm sàng |
   | image | ANOVA chọn 7/10 đặc trưng ảnh |
   | fusion (Φsel, đề xuất) | Chuẩn hoá z, ANOVA 7 ảnh + 9 lâm sàng |
   | fusion_filter | MI + độ quan trọng RF, loại tương quan > 0,95 |
   | fusion_hadamard | SVD hiệp phương sai chéo, d = 5, tích Hadamard |
   | fusion_tensor | Thuật toán 3.5: SVD cắt của M = (1/N)X̃ᵀỸ, ρ = 6 → 12 cột |
   | fusion_wrapper | Tham lam tiến, RF 20 cây, CV 3 phần, 2–7 ảnh và 2–9 lâm sàng |

   Khác bản gốc một điểm: cột lâm sàng còn giá trị thiếu, nên nhánh tensor nhận dữ liệu đã điền trung vị của tập
   huấn luyện như các nhánh khác.
2. **Kịch bản** (`kfold/core.py`, `fkgmm/`): FCM 5 mức fit trên tập huấn luyện; bộ suy diễn chính **FISA-GM-w**
   (`kfold/weighted.py`; định nghĩa đề xuất cho luận án ở `docs/dinh_nghia_FISA_GMw.tex`). M1 FKG-MM vs FKG-UM và
   LR/RF/XGBoost/MLP; M2 Early/Late Fusion; M3 3/5/7 mức; M4 cắt bỏ mờ/đồ thị; M5 năm toán tử fusion; M6 FKGS
   15%/20%; EPS. `"aggregator": "geomean"` cho lại FISA-GM không trọng số.
3. **Lọc luật** (`--rule-base dedup_conf`, `kfold/rulebase.py`): trên cơ sở luật huấn luyện của fold (gồm SMOTE),
   luật trùng giữ 1; tiền đề mâu thuẫn giữ nhãn có độ tin cậy conf(A → l) cao hơn, hai nhãn hoà thì bỏ cả hai.
   **Lưu ý (chạy BRSET 02/11/2026):** AUC giữ nguyên hoặc tăng nhẹ (FKG-MM 92,5 → 92,9) nhưng BalAcc sụp
   (85,3 → 52,2; độ nhạy 87,6% → 4,5%): gộp trùng xoá phần cân bằng lớp do SMOTE tạo ra (còn ~26% luật dương), nên
   ngưỡng s_w > 0 lệch về lớp âm. Chưa dùng BalAcc/Sens./F1 của biến thể này để báo cáo khi chưa hiệu chỉnh ngưỡng.
4. **Báo cáo** (`kfold/report.py`): trung bình ± SD qua 5 fold; Δ, số fold thắng, p của t hiệu chỉnh Nadeau–Bengio
   (dùng kích thước fold thực tế) và Wilcoxon. Mỗi bảng có AUC, **AP** (average precision) và **PR-AUC** (diện tích
   dưới đường Precision–Recall theo quy tắc hình thang) cạnh nhau; mốc của bộ đoán ngẫu nhiên là tỉ lệ dương ≈ 6,6%.
5. **Hình** (`kfold/plots.py`, `<out>/ket_qua/hinh/`): ROC và PR gộp ngoài fold (mỗi ảnh đúng một lần) cho KB-M1/M2
   (`roc_pr_kb_m1_m2`) và KB-M5 (`roc_pr_kb_m5`), AP theo fold (`ap_theo_fold`), dạng PNG 200 dpi và PDF; số liệu gộp
   ở `pooled_oof.csv`. Điểm dự đoán từng ảnh của mọi mô hình: `<out>/ket_qua/predictions/fold_XX.csv.gz`.

Kết quả: `<out>/ket_qua/ket_qua.md`, `per_fold.csv`, `audit_data.csv`, `run_manifest.json`, `chuong3_manifest.json`
(mã băm mã nguồn, dữ liệu, cấu hình, môi trường); `<out>/data/<nhóm>/fold_XX/` (bố cục `KFold_feature_selection_*`).
Số liệu không so trực tiếp với các lần chạy trên bộ 1.529 ảnh hay FRB 12.957 ảnh (khác tập ảnh, khác fold).

## Nguồn

- `fkgmm/`, `kfold/core.py`, `kfold/report.py`: `FKGMM_BRSET3_nested_20261102/ma_nguon`; đã sửa: GM-w và lọc luật
  (tuỳ chọn), Nadeau–Bengio dùng kích thước fold thực tế (bản gốc gán cứng n_tr = 1778, n_te = 242).
- `fusion_strategies.py`: sinh tự động từ `reference/preprocess_source/` (không sửa tay; test so từng hàm).
- `data/`, `build.py`, `portable_folds.py`: gói `FKGE_BRSET_Tan` (kiểm tra mã băm và tái lập fold khi chạy BRSET).
