# Kiểm toán thực nghiệm Chương 3 — FKG-MM trên BRSET 16.266 ảnh

Ngày 03/11/2026. Bộ mã `FKGMM_BRSET_Chuong3` (bản có PR-AUC, hình ROC/PR, lưu dự đoán từng ảnh).
Kết quả: `outputs/chuong3_gmw/ket_qua/` (gói kèm `FKGMM_BRSET_Chuong3_ket_qua_BRSET.zip`).
Số trên tập kiểm tra trong các mục chẩn đoán chỉ dùng **để giải thích**, không dùng để chọn bất cứ thành phần nào.

## 0. Phạm vi và bằng chứng đã kiểm

| Hạng mục | Kết quả |
|---|---|
| Dữ liệu | 16.266 ảnh, 8.524 bệnh nhân, tỉ lệ dương 6,6%; mã băm CSV khớp `data/release_manifest.json`; 5 fold tái lập đúng |
| Chia fold | StratifiedGroupKFold theo bệnh nhân, seed 42; bệnh nhân trùng train/test = 0 ở mọi nhóm, mọi fold |
| Dữ liệu 7 nhóm | 175/175 tệp CSV trùng từng byte giữa hai môi trường độc lập (venv sạch không PyTorch và venv cũ) |
| Mã tiền xử lý | Mọi hàm của `fusion_strategies.py` trùng nguyên văn `Preprocess_kfold_feature_selection.py`; cả 7 nhóm cho kết quả giống hệt quy trình gốc trên ~900 bệnh nhân thật |
| Chống rò rỉ | Đổi toàn bộ tập kiểm tra không làm đổi dữ liệu huấn luyện; SMOTE chỉ thêm dòng dương sau dòng gốc; trọng số GM-w chỉ dùng ảnh gốc của tập huấn luyện |
| FISA | Khớp bản dịch từng dòng của `fisa_module.cpp` (giá trị D và ma trận B) |
| Chỉ số | AP = `average_precision_score`; PR-AUC = quy tắc hình thang; tính lại từ điểm đã lưu khớp `per_fold.csv` đến 1e-12 |
| Tái lập | Hai lần chạy ở hai môi trường: mọi mô hình FKG trùng tuyệt đối; chỉ Random Forest (ảnh) lệch AUC ≤ 8·10⁻⁷ (luồng song song) |
| Test | 18/18 (`tests/test_ch3.py`) |

## 1. Kết luận tổng quan

1. **Quy trình chạy sạch về mặt rò rỉ và tái lập được**, nhưng **khác quy trình mà văn bản luận án (bản v6, Mục 3.6.3–3.6.4) mô tả**:
   không có tập xác thực V, ngưỡng cố định, hàm thuộc FCM ước lượng trên cả mẫu SMOTE, đối chứng không chọn siêu tham số.
   Đây là rủi ro lớn nhất trước hội đồng (phương pháp mô tả ≠ phương pháp sinh số liệu).
2. **FKG-MM không vượt FKG-UM (Bảng) và Logistic Regression trên bảng theo AUC/AP**. Nguyên nhân gốc: mô thức ảnh
   (đặc trưng GLCM) gần như không mang thông tin (AP 8,8 so với mốc ngẫu nhiên 6,6).
3. **FISA trên cơ sở luật bị rút gọn (S-FKGS, C-FKGS, lọc luật trùng) giữ AUC nhưng làm sụp BalAcc** do mất bội số luật;
   cơ chế đã định lượng (Mục 4, Q3).
4. Toán tử **Hadamard mất thông tin ngay ở bước dung hợp** (không phải do mờ hoá).
5. Theo sơ đồ 7 bước: bước 4 (FCM ≈ FIS) và bước 5 đạt; bước 3, 6, 7 đạt một phần (chi tiết Mục 3).

## 2. Kết quả chính (trung bình ± SD qua 5 fold, %)

| Mô hình | AUC | AP | PR-AUC | Sens. | Spec. | BalAcc | F1 |
|---|---|---|---|---|---|---|---|
| **FKG-MM (đề xuất, FISA-GM-w)** | 92,5 ± 1,3 | 53,6 ± 5,5 | 53,5 ± 5,5 | 87,6 ± 1,8 | 82,9 ± 1,0 | 85,3 ± 1,1 | 40,7 ± 1,3 |
| FKG-UM (Bảng) | 92,7 ± 1,4 | 55,0 ± 6,0 | 55,6 ± 6,1 | 91,9 ± 1,5 | 75,6 ± 1,4 | 83,7 ± 0,8 | 34,1 ± 1,4 |
| FKG-UM (Ảnh) | 58,8 ± 2,3 | 8,8 ± 0,7 | 8,6 ± 0,7 | 57,8 ± 7,7 | 57,0 ± 3,3 | 57,4 ± 2,3 | 15,0 ± 1,0 |
| Logistic Regression (bảng) | 92,8 ± 1,1 | 57,5 ± 6,0 | 57,5 ± 6,1 | 92,2 ± 1,6 | 77,6 ± 0,3 | 84,9 ± 0,8 | 36,1 ± 1,9 |
| XGBoost (bảng) | 92,2 ± 1,1 | 47,2 ± 5,0 | 47,7 ± 5,0 | 83,6 ± 4,0 | 88,4 ± 0,7 | 86,0 ± 2,1 | 48,0 ± 2,0 |
| Early Fusion (MLP) | 91,3 ± 2,4 | 50,4 ± 5,3 | 50,2 ± 5,4 | 70,8 ± 8,8 | 92,5 ± 0,6 | 81,6 ± 4,2 | 50,7 ± 3,4 |
| Late Fusion (Ensemble) | 89,7 ± 1,8 | 42,5 ± 4,1 | 42,2 ± 4,1 | 68,0 ± 5,5 | 92,3 ± 1,0 | 80,1 ± 2,3 | 49,0 ± 1,4 |
| Crisp KG (không mờ) | 92,5 ± 1,3 | 54,0 ± 5,2 | 54,0 ± 5,3 | 89,8 ± 1,6 | 81,2 ± 1,1 | 85,5 ± 1,3 | 39,3 ± 1,9 |
| Mờ, không đồ thị (LR trên nhãn ngôn ngữ) | 93,6 ± 1,2 | 57,5 ± 6,2 | 57,4 ± 6,2 | 88,3 ± 5,1 | 85,6 ± 0,6 | 87,0 ± 2,3 | 44,9 ± 2,0 |

So sánh cặp chính (Δ điểm %, số fold thắng, p của t hiệu chỉnh Nadeau–Bengio):
FKG-MM − FKG-UM (Bảng): AUC −0,2 (2/5, p = 0,56), AP −1,5 (1/5, p = 0,37), BalAcc +1,6 (5/5, p = 0,057);
FKG-MM − LR (bảng): AP −3,9 (0/5, p = 0,14); FKG-MM − Late Fusion: AUC +2,8 (5/5, p = 0,043), AP +11,1 (5/5, p = 0,044).
Năm toán tử (AUC / AP): Φsel 92,5 / 53,6; Filter 92,5 / 53,0; Tensor 92,4 / 52,6; Wrapper 91,0 / 47,6; Hadamard 69,9 / 17,8.

Hình: `hinh/roc_pr_kb_m1_m2` (ROC và PR, KB-M1/M2), `hinh/roc_pr_kb_m5` (năm toán tử), `hinh/ap_theo_fold` (AP từng fold);
số liệu gộp ngoài fold ở `hinh/pooled_oof.csv`. Đường cong gộp điểm của 5 fold (mỗi ảnh một lần); vì điểm FISA không được
hiệu chỉnh giữa các fold, AP gộp (FKG-MM 0,523) khác AP trung bình theo fold (0,536) — luận án cần nêu rõ cách tính.

## 3. Đối chiếu với sơ đồ quy trình 7 bước

| Bước trong sơ đồ | Cài đặt thực tế | Bằng chứng | Đánh giá |
|---|---|---|---|
| (1–2) Thu thập, liên kết dữ liệu | Nhãn BRSET + đặc trưng ảnh trích sẵn, ghép theo `image_id` | `build.py`, mã băm release | Đạt |
| (3) Tiền xử lý **bảng** | Điền trung vị, MinMax/chuẩn hoá z, chọn MI/ANOVA — chỉ trên tập huấn luyện của fold | Test so với bản gốc | Đạt |
| (3) Tiền xử lý **ảnh** | 10 đặc trưng GLCM/thống kê trích sẵn (`images_ft.csv`); mã trích xuất và ảnh gốc không có trong gói | `release_manifest.json` (mã băm bộ trích xuất cũ) | **Một phần** — không tái lập được trong gói (Q6) |
| (3) Tiền xử lý **văn bản, video** | BRSET không có | – | Không áp dụng; cần ghi rõ phạm vi |
| (3) Dung hợp đặc trưng | 5 chiến lược: Φsel, Filter, Hadamard, Tensor (Thuật toán 3.5, ρ = 6), Wrapper | Trích nguyên văn, 175 CSV trùng | Đạt; Hadamard mất thông tin (Q4) |
| (chưa có trong sơ đồ) Cân bằng lớp | BorderlineSMOTE trên tập huấn luyện; 46,4% luật sinh từ mẫu tổng hợp | `audit_data.csv` | Thiếu trong sơ đồ (C1) |
| (4) Sinh luật mờ | FCM 5 mức cho mọi khái niệm (coi là thay thế tương đương FIS); mỗi dòng huấn luyện sinh một luật; không có luật chuyên gia, MC-FIS | `kfold/core.py` | Đạt, với điều kiện sửa văn bản (Q1) |
| (5) Sinh FKG | Ma trận A, M, B, C của FKG-Pairs | Khớp bản dịch C++ | Đạt |
| (6) Lấy mẫu FKG → FKGS | Chỉ ở M6 (ngẫu nhiên phân tầng, S-FKGS, C-FKGS với 15%, 20%); M1–M5 dùng đồ thị đầy đủ | `fkgmm/sfkgs.py` | **Một phần** — tỉ lệ không chọn trên V; mất bội số luật (Q3, Q7) |
| (7) Phân loại | FISA-GM-w, nhãn = 1 khi s_w > 0 (ngưỡng cố định) | `kfold/weighted.py` | Đạt về cài đặt; chưa có trong định nghĩa luận án (Q2); ngưỡng không chọn trên V (Q3') |
| (7) Ra quyết định | Không đánh giá căn cứ luật hay giải thích ở Chương 3 | – | Chưa thực hiện (C7) |

## 4. Bất thường và vấn đề, theo mức độ

### Nghiêm trọng

**N1. Văn bản quy trình khác quy trình sinh số liệu.**
- *Vị trí:* `chuong3_mmfkge_v6.tex` Mục 3.6.3 (Phân hoạch dữ liệu và kiểm soát rò rỉ), Mục 3.6.4 (Mô hình đối chứng), Thuật toán chuẩn bị dữ liệu.
- *Văn bản viết:* tách 20% bệnh nhân làm tập xác thực V; chia F_c = 5 phần; SMOTE **trong từng phần**; FCM ước lượng **trên dòng gốc**;
  ngưỡng (β), nhiệt độ, siêu tham số chọn trên V; đối chứng chọn bằng 12 cấu hình ngẫu nhiên theo AUC trên V, ngưỡng theo BalAcc trên V;
  đối chứng ảnh dùng ResNet-50 (PCA 128), chú ý chéo phương thức.
- *Mã đã chạy:* không có V; SMOTE trên toàn bộ tập huấn luyện; FCM trên tập huấn luyện gồm SMOTE; ngưỡng FISA cố định (s_w > 0, tức β = 0);
  đối chứng dùng siêu tham số cố định và ngưỡng 0,5; đặc trưng ảnh thủ công, không có ResNet-50 hay chú ý chéo.
- *Ảnh hưởng:* hội đồng đối chiếu mã với văn bản sẽ thấy hai quy trình khác nhau; mọi bảng của Chương 3 phụ thuộc vào điểm này.
- *Phương án ưu tiên (A):* viết lại Mục 3.6.3–3.6.4 cho đúng giao thức đã chạy. Giao thức này hợp lệ và không rò rỉ (đã kiểm), số liệu không đổi.
  *Phương án B:* cài giao thức có V vào mã (cũng sửa được Q3), sau đó chạy lại toàn bộ.
- *Kiểm chứng cần bổ sung:* nếu chọn B, chạy lại M1–M6 và so với bảng hiện tại.

**N2. FKG-MM không vượt mô hình đơn mô thức trên bảng theo AUC/AP — vì mô thức ảnh gần như không mang thông tin.**
- *Bằng chứng:* FKG-MM − FKG-UM (Bảng): AUC −0,2, AP −1,5 (thắng 1/5). LR (bảng) có AP 57,5 > FKG-MM 53,6 ở cả 5/5 fold.
  FKG-UM (Ảnh) AP 8,8 ≈ mốc 6,6; LR trên 7 đặc trưng ảnh liên tục cũng chỉ đạt AUC 0,584 (fold 1), nên lỗi không nằm ở mờ hoá.
  Ngoài ra 7 cột do người đọc ảnh chấm (`optic_disc`, `vessels`, `macula`, `focus`, `Illuminaton`, `image_field`, `quality`) nằm trong
  mô thức "bảng", khiến FKG-UM (Bảng) mạnh hơn mức một bảng lâm sàng thuần tuý.
- *Ảnh hưởng:* KB-M1 không ủng hộ khẳng định "FKG-MM vượt đơn mô thức" về khả năng xếp hạng.
- *Phương án:* (i) viết lại kết luận KB-M1: tương đương FKG-UM (Bảng) về AUC/AP, nhỉnh hơn về BalAcc (5/5 fold, p = 0,057), vượt rõ
  mô thức ảnh và Late Fusion; (ii) nêu rõ vai trò của 7 cột người đọc ảnh; (iii) cải thiện thật sự cần đặc trưng ảnh mạnh hơn
  (CNN trên ảnh gốc — ngoài phạm vi gói này; máy hiện thiếu 1.758 ảnh gốc).

### Quan trọng

**Q1. Định nghĩa phân hoạch mờ mâu thuẫn với cài đặt đã chọn (FCM 5 mức).**
- *Vị trí:* Định nghĩa dn:mm.fuzzy ("|Λ_j| là số tâm cụm phân biệt thực có (|Λ_j| = 2 với khái niệm nhị phân)") và Mục 3.6.3 ("ước lượng hàm thuộc FCM (trên dòng gốc)").
- *Thực tế:* FCM ước lượng trên tập huấn luyện gồm mẫu SMOTE nên mọi khái niệm có 5 mức; 10/13 cột nhị phân của nhóm bảng có 5 mức,
  nhưng trên ảnh thật chỉ dùng ≤ 2 mức — **3 mức giữa chỉ chứa mẫu tổng hợp** (ví dụ một mức "giới tính ≈ 0,5" không có nghĩa lâm sàng).
- *Quyết định đã có:* giữ FCM 5 mức. Khi đó **phải sửa văn bản**; đề xuất ở Mục 5. Tuỳ chọn `--fcm-fit original` (ước lượng trên dòng gốc)
  có sẵn trong mã để đối chiếu nếu hội đồng hỏi; ở lần kiểm toán trước, việc giữ 2 mức cho biến nhị phân chỉ làm AUC dao động ±0,3.

**Q2. FISA-GM-w chưa có trong định nghĩa của luận án.**
- Định nghĩa dn:mm.fisa chỉ có FISA-P và FISA-GM. Cần thêm định nghĩa FISA-GM-w (`docs/dinh_nghia_FISA_GMw.tex`) và công bố, như
  Nhận xét nx:mm.fisagm đã làm với FISA-GM, rằng biến thể được đưa vào sau khi kiểm toán kết quả ban đầu.

**Q3. Mất bội số luật khi rút gọn cơ sở luật làm sụp BalAcc (S-FKGS, C-FKGS, lọc luật trùng).**
- *Bằng chứng:* S-FKGS 20%: BalAcc 74,3 (Sens. 52); C-FKGS 20%: 72,1 (Sens. 47); lọc luật trùng/mâu thuẫn: 52,2 (Sens. 4,5) —
  so với 85,3 của đồ thị đầy đủ, trong khi AUC giữ 92,2–92,9. Lấy mẫu ngẫu nhiên phân tầng (giữ dòng) không bị ảnh hưởng (85,4).
- *Cơ chế (fold 1, nhóm fusion):* luật dương — phần lớn sinh từ SMOTE — lặp trung bình 6,6 lần (12.134 dòng / 1.827 luật duy nhất),
  luật âm chỉ 2,1 lần (12.134 / 5.700). S-FKGS, C-FKGS (`RuleSampler` trả về một đại diện cho mỗi luật duy nhất) và lọc luật trùng xoá
  bội số này, nên lớp dương mất trọng số gấp khoảng 3 lần lớp âm và điểm FISA lệch về lớp âm. Ở mức 20%, ngân sách mỗi lớp (2.427)
  còn lớn hơn số luật dương duy nhất (1.827), nên tỉ lệ luật dương giảm còn 42,9%.
- *Ảnh hưởng:* KB-M6 chỉ đúng với AUC/AP; BalAcc, độ nhạy và F1 của FKGS không phản ánh chất lượng của phép rút gọn.
- *Phương án ưu tiên:* giữ bội số làm trọng số của luật đại diện (FISA cộng C theo trọng số), sao cho tổng trọng số mỗi lớp bằng
  trước khi rút gọn. *Thay thế:* chọn ngưỡng trên tập xác thực (cần V, xem N1). *Kiểm chứng:* chạy lại M6 với bội số.

**Q3'. Ngưỡng quyết định cố định, không chọn trên tập xác thực.**
- FKG-MM dự đoán dương cho 21,8% ảnh (thực tế 6,6%); precision 26,5% (khoảng 3/4 cảnh báo là dương tính giả); F1 40,7.
  BalAcc tại ngưỡng 0 (0,853) gần mức tối đa đạt được trên cùng đường ROC (0,862), nên ngưỡng phù hợp nếu mục tiêu là sàng lọc
  (ưu tiên độ nhạy); văn bản cần nói rõ mục tiêu này. Nên báo thêm chỉ số độc lập ngưỡng có ý nghĩa lâm sàng: độ nhạy tại độ
  đặc hiệu 90%, precision tại recall 80%.

**Q4. Toán tử Hadamard mất thông tin ngay ở bước dung hợp.**
- LR trên 5 đặc trưng Hadamard liên tục chỉ đạt AUC 0,550, AP 0,066 (≈ ngẫu nhiên; fold 1), trong khi Φsel và Tensor đạt 0,935.
  Cách cài (tích từng thành phần của hai véc-tơ chiếu chuẩn hoá theo dòng) chỉ giữ tương tác nhân, không giữ hiệu ứng chính của
  từng mô thức. So sánh M5 vì thế không công bằng với Hadamard.
- *Phương án:* dùng [x̃P_img ‖ ỹP_tab ‖ x̃P_img ⊙ ỹP_tab] (giữ hiệu ứng chính, như biến thể cross-SVD cũ), hoặc ghi rõ hạn chế trong luận án.

**Q5. Thuộc tính gần hằng lọt vào tập khái niệm.**
- `insuline` thiếu 89,5% và được điền trung vị, nên 95,3% ảnh kiểm tra có cùng giá trị; `vessels` (95,6%), `focus`, `Illuminaton` cũng
  chỉ xuất hiện 1 mức trên tập kiểm tra. `insuline` và `vessels` nằm trong Φsel; các bộ ba chứa chúng gần như không mang thông tin
  (GM-w giảm trọng số, nhưng không loại hết).
- *Phương án:* ấn định trước một bộ lọc biến gần hằng (tỉ lệ giá trị trội > 95% trên tập huấn luyện), áp dụng trước bước chọn đặc trưng;
  hoặc thêm chỉ báo "thiếu" thay cho điền trung vị với biến thiếu trên 80%.

**Q6. Bước 3 chỉ bao phủ 2/4 mô thức; tiền xử lý ảnh không tái lập được trong gói.**
- Văn bản, video: không có trong BRSET. Ảnh: dùng đặc trưng trích sẵn; mã trích xuất (Thuật toán xử lý ảnh trong chương: vùng quan tâm
  theo ngưỡng Otsu, GLCM) và ảnh gốc không nằm trong gói, nên **chưa kiểm chứng** được đặc trưng đã dùng có đúng thuật toán mô tả không.
- *Phương án:* ghi rõ phạm vi (bảng + ảnh), nguồn gốc và mã băm đặc trưng ảnh; nếu cần, đưa bộ trích xuất vào gói và chạy lại trên ảnh gốc.

**Q7. FKGS trong thực nghiệm khác yêu cầu của chính luận án và khác sơ đồ.**
- Luận án yêu cầu "mọi cấu hình FKGS phải được chọn trên tập xác thực" và chỉ dùng khi chi phí vượt ngân sách; M6 dùng tỉ lệ cố định 15%,
  20%. Sơ đồ vẽ bước 6 như bước bắt buộc trước phân loại, trong khi M1–M5 phân loại trên đồ thị đầy đủ.
- *Phương án:* trình bày M6 là phân tích độ nhạy; trên sơ đồ, vẽ bước 6 bằng nét đứt hoặc ghi "tuỳ chọn".

### Cần cải thiện

- **C1.** Sơ đồ thiếu hai bước có thật trong quy trình: cân bằng lớp (SMOTE, 46,4% luật từ mẫu tổng hợp) và ước lượng trọng số bộ ba của GM-w.
- **C2.** Mờ hoá không đem lại lợi ích so với rời rạc hoá đều: Crisp KG = FKG-MM (AUC 92,5 so với 92,5; AP 54,0 so với 53,6; p = 0,94).
  Điều này nhất quán với việc thực nghiệm dùng độ thuộc cứng (dn:mm.fuzzy). KB-M4 không ủng hộ khẳng định về vai trò của mờ; nên viết đúng như vậy.
- **C3.** Đồ thị chưa thêm giá trị dự đoán so với LR trên chính các nhãn ngôn ngữ (AUC 93,6 so với 92,5, thua 5/5 fold; AP 57,5 so với 53,6).
  Lợi thế còn lại của FKG-MM là tính diễn giải (luật, căn cứ), nên cần được đánh giá trực tiếp (C7).
- **C4.** Fold 3 có AP thấp bất thường (0,443 so với 0,53–0,59 ở các fold khác) dù AUC 0,906; fold này có nhiều ca dương nhất (239).
  Nên kiểm tra phân bố (chất lượng ảnh, nguồn khám) để giải thích độ dao động.
- **C5.** AP gộp ngoài fold khác AP trung bình theo fold (0,523 so với 0,536) vì điểm FISA không được hiệu chỉnh giữa các fold; luận án nên
  báo một cách tính chính (theo fold) và ghi chú cách tính của hình.
- **C6.** FKG không lưu nguồn gốc luật (mẫu thật hay tổng hợp), trong khi luận án yêu cầu để tách khi trình bày giải thích.
- **C7.** Bước "ra quyết định" của sơ đồ chưa có đánh giá: không đo chất lượng căn cứ luật hay giải thích ở Chương 3.
- **C8.** Văn bản lỗi thời: `chuong3_v6_muc31_35.tex` dòng 120 (AUC 0,790 so với 0,825 — số của bộ dữ liệu cũ); danh sách "Chưa chạy trong chương" vẫn liệt kê
  mờ không đồ thị, Φ_ten, Φ_had, Φ_wra, nhưng các mục này nay đã chạy.
- **C9.** Lực thống kê thấp: 5 fold, một hạt giống chia fold; Wilcoxon với n = 5 có p nhỏ nhất 0,0625, nên không thể đạt < 0,05.
  Dựa vào t hiệu chỉnh Nadeau–Bengio và nêu rõ giới hạn này; cân nhắc lặp nhiều hạt giống chia fold.

### Hoàn thiện

- **H1.** Random Forest (ảnh) không tất định tuyệt đối giữa hai môi trường (AUC lệch ≤ 8·10⁻⁷, do `n_jobs = -1`); không ảnh hưởng kết luận.
- **H2.** Accuracy của mọi mô hình FKG thấp hơn bộ luôn đoán âm (93,4%); báo cáo đã ghi Acc là độ đo phụ — giữ nguyên.

## 5. Đề xuất sửa văn bản (khớp cài đặt đã chọn)

- **dn:mm.fuzzy:** thay "|Λ_j| là số tâm cụm phân biệt thực có (|Λ_j| = 2 với khái niệm nhị phân)" bằng: "Thực nghiệm dùng Λ = 5 mức cho
  mọi khái niệm; hàm thuộc FCM được ước lượng trên các dòng huấn luyện của fold, gồm cả mẫu tổng hợp do cân bằng lớp. Với khái niệm
  nhị phân, ảnh thật chỉ nhận hai mức ở hai đầu; các mức giữa chỉ xuất hiện ở mẫu tổng hợp."
- **Mục 3.6.3:** viết lại theo giao thức đã chạy: 5 fold theo bệnh nhân; trên tập huấn luyện của fold: điền thiếu, chuẩn hoá, chọn khái
  niệm (trên dòng gốc) → BorderlineSMOTE (k = 5) → FCM → sinh luật → FKG; trọng số GM-w ước lượng chéo theo bệnh nhân trên ảnh gốc;
  quyết định theo s_w > 0; mọi chỉ số chỉ tính trên ảnh thật của fold kiểm tra. Bỏ các câu về V, F_c phần và chọn ngưỡng trên V (hoặc
  giữ lại nếu chọn phương án B của N1).
- **Mục 3.6.4:** đối chứng dùng siêu tham số cố định (liệt kê), ngưỡng 0,5, đặc trưng ảnh thủ công; chuyển ResNet-50 và chú ý chéo
  sang mục "chưa chạy".
- **nx:mm.fisagm:** bổ sung câu công bố cho FISA-GM-w (mẫu ở cuối `docs/dinh_nghia_FISA_GMw.tex`).
- **KB-M1, KB-M4, KB-M6:** viết lại kết luận theo N2, C2–C3, Q3.

## 6. Checklist trước khi chốt Chương 3

- [ ] Chọn phương án cho N1 (sửa văn bản hay cài V) và đồng bộ Mục 3.6.3–3.6.4.
- [ ] Thêm định nghĩa FISA-GM-w và câu công bố (Q2).
- [ ] Sửa dn:mm.fuzzy cho FCM 5 mức (Q1).
- [ ] Viết lại kết luận KB-M1, KB-M4, KB-M6 theo số liệu hiện tại.
- [ ] Quyết định có sửa Hadamard (Q4), lọc biến gần hằng (Q5), giữ bội số trong FKGS (Q3) hay chỉ ghi hạn chế; nếu sửa, chạy lại.
- [ ] Bổ sung bước cân bằng lớp và trọng số GM-w vào sơ đồ; vẽ bước 6 là tuỳ chọn (C1, Q7).
- [ ] Đưa hình ROC/PR và AP theo fold vào chương; ghi chú cách tính đường cong gộp (C5).
- [ ] Cập nhật số liệu lỗi thời và danh sách "chưa chạy" (C8).

## 7. Tái lập

```bash
python -m pip install -r requirements.txt
python -B -X utf8 -m unittest tests.test_ch3 -v
python -B -X utf8 -u chuong3.py all --out outputs/chuong3          # dữ liệu + M1–M6, EPS, hình (khoảng 35 phút CPU)
```
Hoặc trên Colab: `CHAY_CHUONG3_COLAB.ipynb`. Lần chạy dùng cho báo cáo này: `chuong3.py run --out outputs/chuong3_gmw --data outputs/verify/data`,
venv chỉ cài `requirements.txt` (numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, imbalanced-learn 0.14.2, xgboost 3.4.1, matplotlib 3.10.8).
