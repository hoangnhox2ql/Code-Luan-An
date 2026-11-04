# FKG-MM với FISA-GM-w (trung bình nhân có trọng số bộ ba) – 5 fold theo bệnh nhân

Cơ sở luật FKG: luật trùng giữ 1; mâu thuẫn giữ nhãn có độ tin cậy cao hơn (hoà: drop); hàm thuộc FCM ước lượng trên mọi dòng huấn luyện (gồm SMOTE).

5 fold; bệnh nhân trùng train/test = 0; Acc của bộ luôn đoán âm = 93.4% (tỉ lệ dương ≈ 6.6%) – Acc chỉ là độ đo phụ.
AP = average precision (sklearn); PR-AUC = diện tích dưới đường Precision–Recall theo quy tắc hình thang; mốc của bộ đoán ngẫu nhiên cho cả hai là tỉ lệ dương ≈ 6.6%.

## Bảng 7: So sánh tổng hợp với các phương pháp cơ sở

| Method | Modality | Acc | F1 | AUC | AP | PR-AUC | Sens. | Spec. | BalAcc |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Tabular | 78.5 ± 0.3 | 36.1 ± 1.9 | 92.8 ± 1.1 | 57.5 ± 6.0 | 57.5 ± 6.1 | 92.2 ± 1.6 | 77.6 ± 0.3 | 84.9 ± 0.8 |
| Random Forest | Tabular | 89.1 ± 0.7 | 47.3 ± 2.8 | 91.8 ± 1.2 | 41.9 ± 6.1 | 42.2 ± 6.1 | 74.4 ± 5.2 | 90.2 ± 0.6 | 82.3 ± 2.6 |
| XGBoost | Tabular | 88.1 ± 0.8 | 48.0 ± 2.0 | 92.2 ± 1.1 | 47.2 ± 5.0 | 47.7 ± 5.0 | 83.6 ± 4.0 | 88.4 ± 0.7 | 86.0 ± 2.1 |
| MLP | Tabular | 87.2 ± 1.0 | 44.3 ± 1.7 | 90.0 ± 2.3 | 46.5 ± 8.3 | 46.6 ± 8.4 | 77.4 ± 4.2 | 87.9 ± 0.8 | 82.7 ± 2.3 |
| MLP (đặc trưng ảnh) | Image | 74.5 ± 3.8 | 17.3 ± 1.6 | 64.9 ± 1.1 | 11.9 ± 2.0 | 11.7 ± 1.9 | 40.4 ± 5.7 | 76.9 ± 4.3 | 58.6 ± 1.3 |
| Early Fusion (MLP) | Multimodal | 91.0 ± 0.3 | 50.7 ± 3.4 | 91.3 ± 2.4 | 50.4 ± 5.3 | 50.2 ± 5.4 | 70.8 ± 8.8 | 92.5 ± 0.6 | 81.6 ± 4.2 |
| Late Fusion (Ensemble) | Multimodal | 90.7 ± 0.7 | 49.0 ± 1.4 | 89.7 ± 1.8 | 42.5 ± 4.1 | 42.2 ± 4.1 | 68.0 ± 5.5 | 92.3 ± 1.0 | 80.1 ± 2.3 |
| Crisp KG (không mờ) | Multimodal (KG) | 94.4 ± 0.8 | 34.5 ± 3.5 | 92.5 ± 1.6 | 53.8 ± 6.4 | 53.7 ± 6.5 | 22.4 ± 2.3 | 99.4 ± 0.3 | 60.9 ± 1.2 |
| FKG-UM (Ảnh) | Unimodal FKG | 92.6 ± 0.8 | 5.1 ± 1.8 | 60.3 ± 2.2 | 10.2 ± 0.9 | 10.0 ± 0.9 | 3.0 ± 1.4 | 98.9 ± 0.7 | 51.0 ± 0.5 |
| FKG-UM (Bảng) | Unimodal FKG | 93.9 ± 1.0 | 51.5 ± 3.1 | 92.9 ± 1.3 | 54.6 ± 7.8 | 55.2 ± 7.9 | 48.8 ± 3.1 | 97.1 ± 0.9 | 73.0 ± 1.5 |
| **FKG-MM (đề xuất)** | **Multimodal FKG** | 93.6 ± 0.5 | 8.3 ± 5.4 | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 4.5 ± 3.0 | 99.9 ± 0.1 | 52.2 ± 1.5 |

_ResNet-50, EfficientNet-B0, Joint Fusion, Cross-modal Attention cần ảnh gốc – không có trong bộ dữ liệu này._

## Bảng 8: Ablation study mở rộng của FKG-MM

| Variant | Image | Tabular | Fuzzy | Graph | Accuracy | AUC | AP | PR-AUC | BalAcc | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Image-only | ✓ | – | ✓ | ✓ | 92.6 ± 0.8 | 60.3 ± 2.2 | 10.2 ± 0.9 | 10.0 ± 0.9 | 51.0 ± 0.5 | 5.1 ± 1.8 |
| Tabular-only | – | ✓ | ✓ | ✓ | 93.9 ± 1.0 | 92.9 ± 1.3 | 54.6 ± 7.8 | 55.2 ± 7.9 | 73.0 ± 1.5 | 51.5 ± 3.1 |
| Multimodal w/o FKG (Early Fusion MLP) | ✓ | ✓ | – | – | 91.0 ± 0.3 | 91.3 ± 2.4 | 50.4 ± 5.3 | 50.2 ± 5.4 | 81.6 ± 4.2 | 50.7 ± 3.4 |
| Multimodal + Graph, không mờ (Crisp KG) | ✓ | ✓ | – | ✓ | 94.4 ± 0.8 | 92.5 ± 1.6 | 53.8 ± 6.4 | 53.7 ± 6.5 | 60.9 ± 1.2 | 34.5 ± 3.5 |
| Multimodal + Mờ, không đồ thị | ✓ | ✓ | ✓ | – | 85.8 ± 0.5 | 93.6 ± 1.2 | 57.5 ± 6.2 | 57.4 ± 6.2 | 87.0 ± 2.3 | 44.9 ± 2.0 |
| **Full FKG-MM** | ✓ | ✓ | ✓ | ✓ | 93.6 ± 0.5 | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 52.2 ± 1.5 | 8.3 ± 5.4 |

## Trả lời câu hỏi thực nghiệm (AUC là độ đo chính; BalAcc bổ sung)

### KB-M1 – FKG-MM có vượt đơn mô thức?

- FKG-MM − FKG-UM (Bảng): AUC Δ = +0.0; thắng 2/5; p = 0.970; p_W = 0.812; AP Δ = +0.6; thắng 3/5; p = 0.639; p_W = 0.625; BalAcc Δ = -20.8; thắng 0/5; p = 0.000; p_W = 0.062
- FKG-MM − FKG-UM (Ảnh): AUC Δ = +32.6; thắng 5/5; p = 0.000; p_W = 0.062; AP Δ = +45.0; thắng 5/5; p = 0.000; p_W = 0.062; BalAcc Δ = +1.2; thắng 4/5; p = 0.353; p_W = 0.188
- FKG-MM − Logistic Regression (Tabular): AUC Δ = +0.0; thắng 2/5; p = 0.955; p_W = 0.812; AP Δ = -2.3; thắng 1/5; p = 0.165; p_W = 0.125; BalAcc Δ = -32.7; thắng 0/5; p = 0.000; p_W = 0.062
- FKG-MM − MLP (Tabular): AUC Δ = +2.9; thắng 5/5; p = 0.051; p_W = 0.062; AP Δ = +8.8; thắng 5/5; p = 0.007; p_W = 0.062; BalAcc Δ = -30.5; thắng 0/5; p = 0.000; p_W = 0.062
- FKG-MM − MLP (Image): AUC Δ = +28.0; thắng 5/5; p = 0.000; p_W = 0.062; AP Δ = +43.3; thắng 5/5; p = 0.001; p_W = 0.062; BalAcc Δ = -6.4; thắng 0/5; p = 0.006; p_W = 0.062

### KB-M2 – cấu trúc FKG so với hợp nhất không dùng FKG

- FKG-MM − Early Fusion (MLP): AUC Δ = +1.6; thắng 4/5; p = 0.141; p_W = 0.125; AP Δ = +4.8; thắng 5/5; p = 0.088; p_W = 0.062; BalAcc Δ = -29.4; thắng 0/5; p = 0.001; p_W = 0.062
- FKG-MM − Late Fusion (Ensemble): AUC Δ = +3.2; thắng 5/5; p = 0.011; p_W = 0.062; AP Δ = +12.7; thắng 5/5; p = 0.019; p_W = 0.062; BalAcc Δ = -27.9; thắng 0/5; p = 0.000; p_W = 0.062

### KB-M3 – số mức ngôn ngữ

| Số mức | Acc | AUC | AP | PR-AUC | BalAcc | Sens. |
|---|---|---|---|---|---|---|
| FKG-MM, 3 mức | 93.6 ± 0.6 | 92.1 ± 1.0 | 47.6 ± 4.2 | 47.7 ± 4.3 | 54.9 ± 6.8 | 10.5 ± 14.5 |
| FKG-MM, 5 mức | 93.6 ± 0.5 | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 52.2 ± 1.5 | 4.5 ± 3.0 |
| FKG-MM, 7 mức | 93.6 ± 0.5 | 92.3 ± 1.4 | 52.6 ± 5.8 | 52.4 ± 5.8 | 52.0 ± 1.1 | 4.0 ± 2.1 |
- FKG-MM, 3 mức − 5 mức: AUC Δ = -0.8; thắng 1/5; p = 0.126; p_W = 0.125
- FKG-MM, 7 mức − 5 mức: AUC Δ = -0.6; thắng 2/5; p = 0.331; p_W = 0.312

### KB-M4 – vai trò của mờ và đồ thị

- Full FKG-MM − Multimodal + Graph, không mờ (Crisp KG): AUC Δ = +0.4; thắng 4/5; p = 0.402; p_W = 0.188; AP Δ = +1.4; thắng 5/5; p = 0.085; p_W = 0.062
- Full FKG-MM − Multimodal + Mờ, không đồ thị: AUC Δ = -0.7; thắng 1/5; p = 0.195; p_W = 0.125; AP Δ = -2.2; thắng 0/5; p = 0.084; p_W = 0.062
- Full FKG-MM − Multimodal w/o FKG (Early Fusion MLP): AUC Δ = +1.6; thắng 4/5; p = 0.141; p_W = 0.125; AP Δ = +4.8; thắng 5/5; p = 0.088; p_W = 0.062

### KB-M5 – toán tử hợp nhất

| Toán tử | Acc | AUC | AP | PR-AUC | BalAcc | Hạng AUC TB |
|---|---|---|---|---|---|---|
| FKG-MM (đề xuất) | 93.6 ± 0.5 | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 52.2 ± 1.5 | 2.0 |
| FKG-MM (Filter) | 93.7 ± 0.6 | 93.0 ± 1.2 | 55.3 ± 6.3 | 55.2 ± 6.3 | 52.6 ± 1.3 | 1.8 |
| FKG-MM (Hadamard) | 93.1 ± 0.7 | 68.0 ± 3.5 | 16.2 ± 2.2 | 16.1 ± 2.2 | 51.2 ± 0.6 | 5.0 |
| FKG-MM (Tensor) | 94.4 ± 0.8 | 92.4 ± 0.9 | 53.8 ± 5.6 | 53.6 ± 5.7 | 70.7 ± 2.5 | 2.6 |
| FKG-MM (Wrapper) | 91.7 ± 5.6 | 90.9 ± 3.0 | 45.4 ± 11.4 | 45.6 ± 10.8 | 63.4 ± 9.2 | 3.6 |

Friedman p = 0.0076; Kendall's W = 0.70.

### KB-M6 – rút gọn bằng FKGS

| Phương pháp | Tỉ lệ luật | AUC | AP | PR-AUC | BalAcc | Rep | Thời gian chọn + xây (s) | So với đầy đủ (AUC) |
|---|---|---|---|---|---|---|---|---|
| FKG-MM đầy đủ | 100% luật | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 52.2 ± 1.5 | – | 0.874 | — |
| Ngẫu nhiên phân tầng | 15% luật | 92.2 ± 1.2 | 52.0 ± 5.9 | 51.9 ± 5.9 | 56.6 ± 1.6 | 0.907 | 0.575 | Δ = -0.6; thắng 0/5; p = 0.061; p_W = 0.062 |
| S-FKGS | 15% luật | 92.7 ± 1.4 | 53.7 ± 6.6 | 53.5 ± 6.6 | 58.0 ± 2.4 | 0.924 | 6.523 | Δ = -0.2; thắng 3/5; p = 0.507; p_W = 0.812 |
| C-FKGS | 15% luật | 92.6 ± 1.5 | 55.1 ± 7.4 | 55.0 ± 7.5 | 55.7 ± 2.3 | 0.923 | 3.534 | Δ = -0.3; thắng 0/5; p = 0.228; p_W = 0.062 |
| Ngẫu nhiên phân tầng | 20% luật | 92.4 ± 1.7 | 54.1 ± 6.3 | 54.0 ± 6.3 | 56.0 ± 1.0 | 0.919 | 0.787 | Δ = -0.4; thắng 2/5; p = 0.288; p_W = 0.312 |
| S-FKGS | 20% luật | 92.7 ± 1.4 | 54.7 ± 7.1 | 54.6 ± 7.1 | 52.9 ± 1.9 | 0.935 | 7.035 | Δ = -0.2; thắng 1/5; p = 0.314; p_W = 0.188 |
| C-FKGS | 20% luật | 92.8 ± 1.4 | 55.4 ± 6.5 | 55.2 ± 6.6 | 51.5 ± 0.9 | 0.934 | 4.037 | Δ = -0.1; thắng 1/5; p = 0.325; p_W = 0.312 |

### Độ nhạy theo ε của FISA-GM-w (trung bình nhân có trọng số bộ ba)

| ε | Acc | AUC | AP | PR-AUC | BalAcc |
|---|---|---|---|---|---|
| ε = 1e-12 | 93.6 ± 0.5 | 92.7 ± 1.3 | 55.1 ± 6.2 | 55.0 ± 6.2 | 52.3 ± 1.5 |
| ε = 1e-09 | 93.6 ± 0.5 | 92.9 ± 1.3 | 55.2 ± 6.2 | 55.1 ± 6.3 | 52.2 ± 1.5 |
| ε = 1e-06 | 93.6 ± 0.5 | 93.0 ± 1.3 | 55.3 ± 6.3 | 55.2 ± 6.3 | 52.1 ± 1.5 |
