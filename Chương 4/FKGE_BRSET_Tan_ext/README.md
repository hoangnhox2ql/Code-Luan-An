# Gói riêng cho thầy Hồng Tân chạy FKG-E trên BRSET đầy đủ

Thư mục này chạy độc lập: **16.266 ảnh / 8.524 bệnh nhân**, đủ code, CSV,
5 fold và lệnh xuất bảng/ảnh. Không cần tải ảnh gốc hoặc tìm file trong nhánh khác.
Đầu vào là đặc trưng đã trích trước đây; đây chưa phải kết quả huấn luyện FKG-E mới.
Bảng [CHECKED_FOLD_COUNTS.png](CHECKED_FOLD_COUNTS.png) và CSV cùng tên đã được
xuất từ bước chuẩn bị thật của đủ 5 fold, có phân biệt rõ outer/inner train.

## Cập nhật 02/11/2026: làm sạch cơ sở luật và cấu hình chạy nhanh

Mã bám theo `FKGMM_BRSET3_nested_20261102/ma_nguon/fkge_branch` (bản 02/11); ba file mô hình của bản đó
nằm nguyên vẹn trong `reference/` để đối chiếu số học. Thêm hai yêu cầu đã chốt:

- **Trước khi huấn luyện, làm sạch cơ sở luật** (`training_data` trong `train_fkge.py`): luật có cùng tiền đề
  nhưng khác nhãn (luật mâu thuẫn) bị loại cả hai; mỗi luật còn lại giữ **một** bản (trọng số 1). Fold 1–5:
  loại 3.333 / 5.426 / 4.964 / 3.693 / 2.948 dòng mâu thuẫn (15–28% số dòng gồm SMOTE), còn
  7.433 / 5.021 / 5.721 / 6.999 / 8.409 luật. Khoảng 30% số luật là luật dương (SMOTE sinh nhiều dòng dương trùng).
- **Mẫu huấn luyện** của L_pred/L_inf/L_rule: các dòng (luật, nhóm) của luật còn lại, giữ trọng số số dòng
  (40–44% dương); dòng mâu thuẫn bị loại. Validation và test (ảnh thật) không đổi.
- **Chống rò rỉ khi mỗi luật chỉ còn một bản:** mẫu thuộc nhóm u chỉ thấy luật có ít nhất một dòng từ nhóm
  khác (nếu mọi dòng của luật thuộc u thì −10⁶, đúng `_mask` của nhánh); FISA giáo viên của phần p dùng các
  luật có dòng ngoài p. Cả hai trùng với việc dựng lại cơ sở luật "giữ 1" khi bỏ nhóm u hoặc phần p. Việc loại
  luật mâu thuẫn làm một lần trên toàn inner train.
- **Cấu hình chạy nhanh** (`config.json`): lưới 24 cấu hình = α {0,5; 0,7} × d_e {6, 8, 16} × λ_I {0,1; 0,5} ×
  λ_P {0,3; 0,7}; tối đa 50 vòng (T_ep), patience 10. Baseline KGE dùng cùng d ∈ {6, 8, 16} (`config_extra.json`).
- **Không đổi tổ chức dữ liệu:** `data/`, 5 outer fold, chia inner train/validation theo bệnh nhân, tiền xử lý,
  SMOTE trong từng phần teacher, nhóm theo bệnh nhân + hai mẫu cha SMOTE; `prepared.npz` giữ đúng các khóa
  như bản đầu. Chuẩn bị lại 5 fold cho đúng các cột cũ của `CHECKED_FOLD_COUNTS.csv`.
- Kết quả **không so trực tiếp** với các bản trước (cơ sở luật, FISA và lưới đều đã đổi). Output của bản trước bị
  `extra.py` từ chối; chạy lại `run.py` trên thư mục mới.

## Chạy trên Colab

**Bản này:** tải `FKGE_BRSET_Tan_ext.zip` lên `MyDrive/`, mở
[CHAY_CHUONG4_COLAB.ipynb](CHAY_CHUONG4_COLAB.ipynb) với `MODE = "all"` (lượt chính rồi bổ sung). Output mặc định
`MyDrive/FKGE_BRSET_Tan_ext_v3_16266` là thư mục mới, không trộn với output của các bản trước.

Notebook GitHub dưới đây tải **gói gốc** `FKGE_BRSET_Tan` (cơ sở luật chưa làm sạch, lưới 108), giữ để đối chiếu:
mở [notebook Colab](https://colab.research.google.com/github/thanhst/Fuzzy-Knowledge-Graph/blob/main/deliverables/FKGE_BRSET_Tan/CHAY_TREN_COLAB.ipynb),
chọn GPU trong Runtime rồi chạy lần lượt các ô. Notebook tải đúng gói ZIP riêng,
cài thư viện, kiểm tra dữ liệu và chạy cả 5 fold. Kết quả/checkpoint được lưu
trên Google Drive của người chạy; cache teacher lớn nằm trong ổ tạm của Colab.

Lưới của bản này có **24 cấu hình/fold**, tối đa 50 vòng/cấu hình, early stopping
patience 10. Chưa đo thời gian chạy đầy đủ trên Colab; số đo CPU của một cấu hình trên fold 1 thật nằm
trong `VERIFICATION.md`.
Notebook có `--resume`: chạy lại cùng lệnh sẽ tiếp tục từ cấu hình đã hoàn tất.
Thay code, dữ liệu, seed, môi trường hoặc cấu hình thì cần output mới.

## Chạy trên máy riêng

Mở terminal ngay trong thư mục `FKGE_BRSET_Tan`. Môi trường kiểm thử:
Python 3.13.7, PyTorch 2.11.0+cu130 (chạy CPU), các phiên bản số học trong
`requirements-common.txt`. `requirements.txt` khóa phiên bản PyTorch;
`requirements-colab.txt` dùng PyTorch/CUDA có sẵn của Colab và ghi phiên bản thực tế vào manifest.

```bash
python -m pip install -r requirements.txt
python -B -X utf8 run.py --check
python -B -X utf8 -u run.py --device auto --output outputs/brset_full --resume
```

Có thể dùng môi trường ảo riêng. Trên máy chỉ chạy CPU, cài wheel CPU trước:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Chạy thử nhỏ để xác nhận môi trường, **không tạo kết quả BRSET**:

```bash
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -u run.py --smoke --folds 1 --device cpu --output outputs/smoke --resume
```

Chỉ chuẩn bị dữ liệu và xuất bảng đếm của đủ 5 fold:

```bash
python -B -X utf8 -u run.py --stage prepare --device auto --output outputs/brset_full --resume
```

Sau đó chạy lệnh đầy đủ bên trên, giữ nguyên `--device` và output để tiếp tục.
Nếu RAM GPU hạn chế, dùng `--block-size 64` ngay từ đầu; block chỉ chia phép tính,
không thay gradient toàn bộ hay số mẫu thống kê. Chạy CPU được nhưng chưa có số đo
thời gian đầy đủ. Teacher dùng file memory map trong thư mục tạm, có thể cần vài GiB
ổ đĩa; mất cache tạm thì gói tự tính lại từ dữ liệu chuẩn bị đã lưu.

## Đầu ra gửi lại

Trong `outputs/brset_full/reports/` hoặc thư mục Drive tương ứng:

- `fold_population.csv` và `.png`: ảnh, bệnh nhân ở outer train/inner train/validation/test,
  số dòng luật gồm SMOTE, số dòng mâu thuẫn đã loại, số luật duy nhất còn lại, số mẫu SMOTE, số nhóm teacher
  và nhóm lớn nhất.
- `metrics_by_fold.csv` và `.png`: FISA và FKG-E trên outer test từng fold.
- `metrics_mean_std.csv` và `.png`: trung bình và độ lệch chuẩn mẫu giữa các outer fold.
- `metrics_pooled_oof.csv`, `predictions_oof.csv`, `roc_pr.png`: chỉ số và đường ROC/PR
  từ các dự đoán ngoài fold; mỗi ảnh được đánh giá đúng một lần sau đủ 5 fold.
- `selected_hyperparameters.csv`: alpha, chiều embedding, lambda I/P, validation loss và epoch đã chọn.
- `confusion_fisa.csv`, `confusion_fkge.csv`: ma trận nhầm lẫn.

`average_precision` là AP của scikit-learn; `pr_auc_trapezoidal` là diện tích PR
tính theo hình thang. Báo riêng cả hai, không gọi hai cách tính là cùng một chỉ số.
Mean/std theo fold khác chỉ số pooled OOF. Khi mới chạy một phần fold, báo cáo
ghi đúng số fold hoàn tất; cần đủ 5 fold để báo cáo thí nghiệm này.

Mỗi `fold_XX/` giữ danh sách ID thật, nguồn hai mẫu cha của SMOTE, tiền xử lý,
checkpoint từng cấu hình, lịch sử học, cấu hình được chọn và dự đoán test.
`run_manifest.json` ghi code/commit hoặc thông tin phát hành ZIP, seed, SHA256,
phiên bản thư viện, CPU/GPU/RAM, câu lệnh chính xác và lần resume.
`environment-freeze.txt` ghi môi trường thực tế để tái lập.

## Giao thức đã nối trong gói

1. Outer split giữ nguyên `StratifiedGroupKFold(5, shuffle=True, random_state=42)`
   theo `patient_id` trong CSV đã xác minh; không lấy lại bộ khoảng 1.500 ảnh.
2. Trong mỗi outer train, lấy một phần của SGKF 5 phần làm validation thật,
   seed `42 + fold`; không trùng bệnh nhân giữa inner train, validation và test.
3. Fit median imputer, StandardScaler, chọn 7 đặc trưng ảnh + 9 lâm sàng bằng ANOVA
   chỉ trên inner train. Số đặc trưng, FCM 5 mức và các hằng số trong backend là
   lựa chọn cố định từ trước thí nghiệm, không chọn bằng test.
4. Chia inner train thành 5 phần teacher theo bệnh nhân, rồi chạy BorderlineSMOTE
   riêng trong từng phần. Mẫu tổng hợp và cả hai mẫu cha ở cùng phần; giá trị thiếu,
   scaler và lựa chọn thuộc tính không fit lại trên validation/test. FCM fit trên
   dữ liệu huấn luyện sau SMOTE; teacher của một phần chỉ dùng luật từ bốn phần khác.
5. Giữ nhóm theo bệnh nhân và nguồn SMOTE. Hai bệnh nhân thật có cùng luật mờ
   vẫn là hai bệnh nhân (hai nhóm). Cơ sở luật bỏ luật mâu thuẫn và giữ mỗi luật một lần; luật vẫn nhớ các dòng
   (nhóm, phần) sinh ra nó để giáo viên và mặt nạ FKG-E chỉ bỏ đúng luật mà phần/nhóm đó tự sinh ra.
6. Chọn alpha, d_e, lambda I/P, epoch, ngưỡng và nhiệt độ hiệu chỉnh bằng validation.
   Không dùng outer-test loss/AUC/AP để chọn. Khóa lựa chọn trước khi đánh giá test;
   không refit gộp validation trong giao thức này.
7. Đơn vị đánh giá là 5 outer fold. Các phần teacher, nhóm nối cha SMOTE và block
   tính toán **không được coi là các mẫu độc lập** để tăng n hoặc tính CI.

Backend dùng float64 và gradient toàn bộ có chia block để giảm RAM. Kiểm tra
trong `tests/test_pipeline.py` đối chiếu giá trị FISA, attribution, giáo viên tính chéo, mặt nạ, loss và
gradient FKG-E với code của nhánh Chương 4 trong `reference/` trên cơ sở luật đã làm sạch.
Đây là cùng hàm mục tiêu trên cùng cơ sở luật; kết quả thí nghiệm cũ không được kỳ vọng
khớp vì bộ dữ liệu và giao thức validation/SMOTE đã thay đổi.

`SOURCE_PROVENANCE.json` ghi hash zip nguồn và hai file mô hình gốc giữ nguyên.
`data/release_manifest.json` giữ bằng chứng của đợt xuất CSV ban đầu;
hash code trong manifest dữ liệu là code xuất CSV của lần đó.
`build.py` và `portable_folds.py` dùng kiểm tra/tạo lại CSV, không dùng để huấn luyện.
`run.py` chỉ gồm thí nghiệm FKG-E/FISA chính; KB2–KB6, Ablation và baseline nằm ở `extra.py` (mục dưới).

## Thí nghiệm bổ sung Chương 4: KB2–KB6, Ablation, Baseline (`extra.py`)

Trên Colab: tải `FKGE_BRSET_Tan_ext.zip` lên `MyDrive/` rồi mở [CHAY_CHUONG4_COLAB.ipynb](CHAY_CHUONG4_COLAB.ipynb)
(chạy lượt chính + bổ sung, hoặc `MODE = "extra_only"` trên output lượt chính của chính bản này; log và checkpoint lưu trên Drive).

Chạy **sau** `run.py` của bản này, trên đúng thư mục output đó (output của gói gốc hay bản trước dùng cơ sở luật khác
nên bị từ chối). Mọi kịch bản xuất phát từ cấu hình FKG-E đã khóa trên
validation của từng fold (`fold_XX/selected.json`) và chỉ thay đúng một yếu tố; outer test không dùng để chọn.

```bash
python -B -X utf8 -u extra.py --output outputs/brset_full --device auto --resume
python -B -X utf8 -u extra.py --output outputs/brset_full --only kb3,kb4,baseline --resume  # không huấn luyện FKG-E mới
python -B -X utf8 extra.py --output outputs/brset_full --stage report                       # chỉ xuất lại bảng
```

| Kịch bản | Nội dung | Chi phí thêm / fold |
|---|---|---|
| KB2 | FISA và FKG-E trên cơ sở luật nén S-FKGS θ=0,3 (Chương 2) so với cơ sở luật đầy đủ | 1 teacher + 1 FKG-E |
| KB3 | d_e ∈ {6, 8, 16}: với mỗi giá trị, lấy cấu hình trong lưới 24 có validation loss nhỏ nhất | 0 (đọc checkpoint) |
| KB4 | (λ_I, λ_P) 2×2 và hồ sơ α (2 giá trị), cùng cách trên; ảnh `kb4_heatmap.png` | 0 |
| KB5 | Số mẫu âm K ∈ {2, 5, 10} (K=5 là mô hình chính) | 2 FKG-E |
| KB6 | Thời gian suy diễn theo \|R\|: 20/40/60/80/100% luật (đã làm sạch) lấy ngẫu nhiên không hoàn lại (seed 7) | 4 teacher + 4 FKG-E |
| Ablation | Bỏ L_pred, L_inf, L_rule, cạnh, nút, cả cấu trúc; gộp trung bình đều α=r/(r+1) | 7 FKG-E |
| Baseline | FISA; Node2Vec/TransE/DistMult + kNN trên cùng cơ sở luật; LR, RF, XGBoost trên cùng đặc trưng đã chọn | vài phút |

Ghi chú phương pháp:
- KB2/KB6: tập luật bị nén/lấy mẫu con, còn truy vấn huấn luyện (L_pred, L_inf, L_rule) vẫn là toàn bộ inner
  train, như giao thức lồng nhau cũ. Teacher tính chéo dùng luật của bốn phần còn lại, đúng phân vùng bệnh nhân.
  S-FKGS và KB6 chọn trên cơ sở luật đã làm sạch (n = số luật, ngân sách m = round(θn); KB6 lấy ⌈r·n⌉ luật);
  luật được chọn giữ nguyên các dòng sinh ra nó cho mặt nạ chống rò rỉ. `tests/test_extra.py` kiểm tra S-FKGS
  chọn cùng tập luật, cùng m và Rep với `reference/sfkgs.py`, và loss/gradient trên hai loại tập luật con.
- KB3/KB4 là phân tích độ nhạy của lưới chính; giá trị test của từng điểm **không** dùng để chọn.
- Các biến thể ablation, K, alpha="mean" và tập luật con được đối chiếu loss/gradient với `reference/fkge.py`.
  Mặc định (λ_R=λ_E=λ_N=1, K=5) cho đúng hàm mục tiêu của lượt chính.
- Baseline KGE học trên cùng cơ sở luật đã làm sạch, **mọi** cặp/bộ ba ở mỗi epoch (bản *Lite cũ chỉ 2.000
  mẫu/epoch), d ∈ {6, 8, 16} (cùng mức của FKG-E) và k ∈ {5,25}
  chọn theo validation log loss; luật hòa tại láng giềng thứ k chia phần phiếu còn lại theo trọng số.
  ML cổ điển học trên ảnh thật của inner train (không SMOTE, có trọng số lớp), tham số chọn theo validation
  log loss. Mọi phương pháp chọn ngưỡng bằng balanced accuracy trên validation. Số liệu baseline không so
  trực tiếp với bản cũ.
- Thời gian/truy vấn: một lượt khởi động rồi lấy lần nhanh nhất trong 3 lần, đo trên toàn bộ outer test của fold.
  Chỉ so sánh thời gian giữa các job chạy trên cùng phần cứng (`extra/extra_manifest.json` ghi thiết bị).

Kết quả: `extra/reports/` gồm `*_by_fold.csv`, `*_mean_std.csv|png` (trung bình ± SD mẫu qua outer fold, cột
`n_folds`), `kb4_heatmap.png`, `kb6_time.png`, `baseline_pooled_oof.csv`. `README.txt` đánh dấu kịch bản nào
chưa đủ 5 fold. Mỗi job lưu JSON + dự đoán test trong `extra/jobs/fold_XX/`; chạy lại cùng lệnh với `--resume`
sẽ bỏ qua job đã xong. Khi đánh giá lại, mô hình chính phải cho đúng chỉ số trong `results.json`; nếu lệch,
báo cáo dừng với lỗi.

## Kịch bản bổ sung KB-E6 (03/11/2026): chi phí suy diễn theo số thuộc tính r (`kb_r.py`)

Mục đích: kiểm chứng ràng buộc (R4) cho **FKG-E đầy đủ** theo số thuộc tính r. Chi phí FISA bảng tra tăng theo
C(r,3), còn chi phí FKG-E tăng theo r·d_e + |R|·d_e. Đồng thời đo R1–R3 để xem lợi thế chi phí có phải trả bằng chất lượng.

- **Tăng r như phép hợp nhất đa mô thức:** r = 8, 16, 23 là đặc trưng ảnh và lâm sàng chọn bằng ANOVA (23 = toàn bộ đặc trưng).
  r = 32, 40, 48 thêm 9, 17, 25 đặc trưng **hợp nhất tensor**: tích từng đặc trưng ảnh với từng đặc trưng lâm sàng đã chuẩn
  hoá, chuẩn hoá lại và chọn bằng ANOVA chỉ trên inner train (`FusionPreprocessor(n_tensor=...)`; với `n_tensor = 0`, gói
  không đổi: `prepared.npz` của fold 1, r = 16 trùng từng phần tử với lượt chính).
- **Giao thức:** như `run.py` (tách validation theo bệnh nhân, SMOTE trong từng phần giáo viên, FCM 5 mức, làm sạch cơ sở luật,
  giáo viên FISA tính chéo). Mỗi (r, fold) huấn luyện một FKG-E đầy đủ với tham số cố định trong `config_kb_r.json`
  (α 0,7; d_e 16; λ_I 0,1; λ_P 0,3: cấu hình được chọn nhiều nhất ở lượt chính), tối đa 50 vòng. Trên cùng quỹ đạo đọc
  hai quy tắc dừng chỉ dùng validation: (a) log loss validation nhỏ nhất, patience 10 (= `CompactFKGE.fit`); (b) AUC
  validation lớn nhất, patience 10.
- **Đo:** |R|; R1 (AUC, AP, BalAcc của FISA-GM và FKG-E); R2 (đồng thuận nhãn với FISA); R3 (Dev = trung bình
  ½‖a_E − a_F‖₁); R4: số phép tính mỗi truy vấn (FISA tuần tự, FISA bảng tra, FKG-E; mỗi phép nhân, cộng, so sánh, đọc
  bảng, exp, log là 1), thời gian CPU 1 luồng (theo lô và từng truy vấn; FKG-E với nhúng luật tính sẵn như Thuật toán 4.2,
  FISA bảng tra tách bảng theo nhãn; cả hai được kiểm tra trùng giá trị với cài đặt của gói) và bộ nhớ của mô hình suy diễn.
  Thời gian chỉ so trong cùng lượt chạy trên cùng máy.

```bash
python -B -X utf8 -u kb_r.py --device auto                       # mọi r, 5 fold; chạy lại để tiếp tục
python -B -X utf8 -u kb_r.py --device cuda --r 40 48 --folds 1 2  # một phần lưới
python -B -X utf8 kb_r.py --stage report                         # chỉ xuất lại bảng
python -B -X utf8 -u kb_r.py --smoke --folds 1 2 3 --device cpu --output outputs/kb_r_smoke
```

Kết quả: `outputs/kb_r/reports/` (`kb_r_by_fold.csv`, `kb_r_mean_std.csv`, `kb_r.png`, `README.txt` với tỉ số
T_E/T_seq, T_E/T_tab và tỉ số thời gian cho từng r). Mỗi `r??/fold_??/` giữ dữ liệu chuẩn bị, dự đoán và `result.json`
(gồm quỹ đạo validation). Thêm `kb_r.py` và nhánh tensor làm đổi hash mã của gói, nên output lượt chính và `extra/` tạo bằng
bản trước không chạy tiếp (`--resume`) được bằng bản này; số liệu của chúng không đổi.
