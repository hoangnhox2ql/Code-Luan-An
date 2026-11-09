# Kiểm tra gói trước khi bàn giao

## Bản 02/11/2026 (mới nhất): làm sạch cơ sở luật, cấu hình chạy nhanh, notebook Chương 4

Thay đổi: loại luật mâu thuẫn và giữ mỗi luật trùng một bản trước khi huấn luyện; mẫu huấn luyện là các dòng
(luật, nhóm) còn lại, giữ trọng số số dòng; lưới 24 cấu hình, T_ep tối đa 50, patience 10, KGE d ∈ {6, 8, 16};
notebook `CHAY_CHUONG4_COLAB.ipynb`. Tham chiếu: `reference/*.py` = bản 02/11 của
`FKGMM_BRSET3_nested_20261102/ma_nguon/fkge_branch` (hash trong `SOURCE_PROVENANCE.json`).

- **17/17 test đạt** (`python -m unittest discover -s tests`):
  - làm sạch đúng định nghĩa (so với cách đếm trực tiếp trên các dòng): tập luật, `rows`, trọng số 1, số dòng
    và tiền đề mâu thuẫn bị loại;
  - loss và gradient FKG-E trùng `reference/fkge.py` tới 1e-10 (luật trọng số 1; luật chỉ do một nhóm sinh ra
    mang `unit_counts` của nhóm đó), ở block 2/11/1000, mọi biến thể ablation, K ∈ {2, 10}, gộp trung bình đều,
    tập luật S-FKGS và tập luật lấy mẫu 40%/80%;
  - mặt nạ trùng `FKGE._mask` của tham chiếu và trùng định nghĩa "cơ sở luật dựng lại khi bỏ nhóm của mẫu";
  - giáo viên: luật dùng cho phần p trùng định nghĩa "dựng lại khi bỏ phần p"; D, quyết định và attribution trùng
    `reference/fisa.py` tới 1e-11; attribution bằng 0 trên luật phần đó không được dùng;
  - FISA và attribution trùng tham chiếu; S-FKGS chọn cùng tập luật, cùng m và Rep.
  - Sửa nhỏ: khi hai lớp có số luật bằng nhau và mẫu không khớp nút nào, nhãn dự phòng của `CompactFISA` nay
    lấy theo luật đầu tiên như tham chiếu (trước đó lấy nhãn nhỏ hơn; trên BRSET hai lớp không bao giờ bằng nhau).
- **Tổ chức dữ liệu không đổi:** `data/` trùng từng byte với bản đầu; `run.py --check` đạt; chuẩn bị lại 5 fold
  thật cho đúng 11 cột của bảng đếm bản đầu; `prepared.npz` có đúng các khóa của bản đầu và trùng từng phần tử.
  Sau làm sạch (fold 1–5): loại 3.333 / 5.426 / 4.964 / 3.693 / 2.948 dòng mâu thuẫn (251 / 367 / 325 / 283 / 275
  tiền đề), còn 7.433 / 5.021 / 5.721 / 6.999 / 8.409 luật và 11.093 / 9.962 / 10.219 / 11.026 / 11.579 mẫu.
- **Chạy thật fold 1, một cấu hình (α 0,7; d_e 16; λ_I 0,5; λ_P 0,3), CPU đang tải nặng (load ≈ 30/8 lõi):**
  teacher 131 s; FKG-E dừng sớm ở vòng 36 (tốt nhất vòng 26), 288 s (8,0 s/vòng), không vòng nào chiếu hay lỗi
  tìm kiếm theo đường. Test (cấu hình đơn lẻ, không qua chọn lưới — chỉ để kiểm tra, không báo cáo): FISA
  AUC 0,917, FKG-E AUC 0,924.
- **Smoke đầu-cuối (dữ liệu tổng hợp):** `run.py`, `extra.py` (đủ kịch bản), `--stage report`; `extra.py` từ chối
  output tạo bằng cơ sở luật khác; notebook `CHAY_CHUONG4_COLAB.ipynb` chạy từ zip ở `MODE = "smoke"`.
- **Chưa làm:** chưa chạy đủ lưới 24 cấu hình hay job bổ sung trên BRSET thật; chưa chạy trên GPU/Colab (notebook có
  ô chạy thử GPU trên dữ liệu tổng hợp trước lượt dài).

## Bản 02/11/2026 trước đó: luật gộp có trọng số (đã thay; lưu ở `FKGE_BRSET_Tan_ext_gop_luat_trong_so.zip`)

- Luật gộp (trọng số = số dòng, mặt nạ log(w − c_u)); 18/18 test với tham chiếu `dedup`; trên fold 1 thật trùng
  bản đầu: tham số ≤ 6e-17, loss lệch hằng số L_rule 0,0598.

## Bản đầu (trước khi gộp luật)

- 6 bài kiểm tra đã đạt: giá trị/attribution FISA; loss/gradient FKG-E so với code
  zip nguồn trên luật trùng có trọng số; bất biến theo block; SMOTE giống thư viện
  và giữ đúng hai mẫu cha; teacher không tách bệnh nhân/nguồn SMOTE; validation
  không thay tham số tiền xử lý. Một bài có thể kiểm tra nhiều tính chất.
- Đã chạy chuẩn bị trên đủ 16.266 ảnh, cả 5 fold; bảng số lượng trong
  `CHECKED_FOLD_COUNTS.csv` và `.png`. Không có bệnh nhân giao nhau giữa
  inner train, validation và outer test.
- Đã chạy thử toàn bộ luồng CPU trên dữ liệu tổng hợp: 200 ảnh / 100 bệnh nhân,
  một outer fold, hai cấu hình, tối đa hai epoch; tạo được checkpoint, dự đoán,
  bảng chỉ số và ảnh. Đây là kiểm tra phần mềm, không phải chỉ số BRSET.
- Chưa huấn luyện đủ lưới trên bộ BRSET thật; chưa xác nhận lượt chạy GPU/Colab
  đầy đủ. Môi trường kiểm thử và lệnh chạy nằm trong README; lượt chạy mới tự ghi
  môi trường thực tế, phần cứng, câu lệnh, seed, commit và hash vào output.

## Bổ sung KB2–KB6, Ablation, Baseline (02/11/2026, `extra.py`)

- 12/12 test đạt (6 cũ + 6 mới trong `tests/test_extra.py`): các tùy chọn λ_R/λ_E/λ_N, K, gộp trung bình đều
  và tập luật con có truy vấn đầy đủ khớp loss/gradient với `reference/fkge.py`; teacher với cơ sở luật đầy đủ
  truyền tường minh trùng từng bit; S-FKGS nén chọn cùng tiền đề, cùng Rep với `reference/sfkgs.py`;
  kNN có trọng số bằng kNN trên luật trải phẳng.
- Smoke đầu-cuối trên dữ liệu tổng hợp (3 fold): đủ mọi kịch bản, resume và `--stage report` hoạt động;
  mô hình chính đánh giá lại trùng `results.json`.
- Fold 1 BRSET thật (chỉ prepare, CPU): S-FKGS 12.128 → 5.386 luật, Rep 0,991, 16 s; KGE 5–48 s mỗi d;
  LR/RF/XGBoost 1–14 s. **Chưa chạy** các job FKG-E bổ sung trên BRSET thật.

## KB-E6 (03/11/2026, `kb_r.py`)

- 22/22 test đạt (17 cũ + 5 mới trong `tests/test_kb_r.py`): nhánh tensor không đổi gì khi `n_tensor = 0` (kể cả với
  preprocessor đã pickle trước đó) và chỉ fit trên tập huấn luyện; quy tắc dừng (a) trên quỹ đạo trùng `CompactFKGE.fit`
  (cùng validation loss, cùng tham số); FKG-E tính sẵn, FISA bảng tra tách nhãn và FISA tuần tự cho cùng giá trị với cài
  đặt của gói; công thức đếm phép tính.
- Hồi quy trên BRSET thật: chuẩn bị lại fold 1 với r = 16 bằng mã mới cho `prepared.npz` trùng từng phần tử với lượt chính.
- Smoke `kb_r.py --smoke` (dữ liệu tổng hợp, 3 fold, r = 4 và 6) chạy hết và xuất báo cáo.
