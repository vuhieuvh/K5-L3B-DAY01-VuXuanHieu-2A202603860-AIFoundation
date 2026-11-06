# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603860
- **SEED** 3251

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.989 | 0.778 | 0.871 | 0.857 | 2.080 | 44212.196 | 0.032 | 0.043 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.991 | 0.949 | 0.969 | 0.947 | 8.441 | 20403.972 | 0.985 | 3.434 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.881 | 0.949 | 0.914 | 0.974 | 35.950 | 29.473 | 61.493 | 84.249 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.982 | 0.949 | 0.965 | 0.923 | 36.627 | 73.153 | 25.166 | 39.729 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.982 | 0.923 | 0.952 | 0.947 | 38.378 | 71.460 | 25.500 | 36.402 | 0.0 |

## 3. Dự đoán và thực tế

1. **`dl_curve` (BoW + MLP so với BoW + LogReg):**
   - *Dự đoán:* F1 của MLP sẽ chỉ xấp xỉ bằng hoặc hơn nhẹ so với Logistic Regression (~0.87–0.88).
   - *Thực tế:* BoW + MLP đạt F1 = **0.969**, cao hơn đáng kể so với BoW + LogReg (**0.871**).
   - *Vì sao lệch:* Mạng MLP dùng bộ tối ưu Adam và huấn luyện theo mini-batch qua 30 epoch nên hội tụ tốt hơn nhiều so với 300 bước full-batch Gradient Descent với `lr=0.5` của Phần 1.
2. **`dl_relu` (Bỏ ReLU trong MLP):**
   - *Dự đoán:* Bỏ ReLU sẽ làm F1 giảm nhẹ về mức của Phần 1.
   - *Thực tế:* Khi bỏ ReLU, mạng tuyến tính 2 lớp vẫn đạt F1 = **0.969** (bằng hệt khi có ReLU).
   - *Vì sao lệch:* Trên dữ liệu Bag-of-Words 2.000 từ, ranh giới phân tách tuyến tính đã rất tốt khi được tối ưu bằng Adam; bản thân hàm phi tuyến ReLU không phải yếu tố chính tạo nên mức tăng F1.
3. **`hard_eval` (Stress test trên các mô hình):**
   - *Dự đoán:* Tưởng chỉ có nhóm né từ khóa và prompt injection là khó nhất.
   - *Thực tế:* Nhóm **Spam giọng tự nhiên (`natural_spam`)** làm cả 5 mô hình đều đoán sai hàng loạt (6/7 ca tất cả mô hình cùng sai thuộc nhóm này), và BoW + LogReg tụt F1 từ 0.857 xuống chỉ còn **0.308**.
   - *Vì sao lệch:* Những tin spam viết hoàn toàn giống hội thoại bạn bè đời thường không hề có từ khóa rác và mang ngữ nghĩa giao tiếp cá nhân nên vượt qua cả bộ lọc từ vựng lẫn không gian vector ngữ nghĩa.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** `1. BoW + LogReg` tụt mạnh nhất (F1 từ **0.857** trên SUB xuống **0.308** trên bộ case khó, Accuracy chỉ đạt **50.9%**), đặc biệt thất bại nặng ở nhóm *Né từ khóa (obfuscation)* và *Spam giọng tự nhiên*.
- **2 case cụ thể:**
  1. **Case #31 (`obfuscation`, nhãn thật = spam):** *"U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461"*.
     - *Cơ chế sai:* Chèn dấu chấm giữa từng chữ cái làm bộ tách từ `re.findall(r"[a-z0-9]+", ...)` tách thành các ký tự đơn lẻ (`s`, `e`, `l`...), biến toàn bộ từ khóa spam thành từ ngoài từ điển (OOV) và làm hỏng luôn biểu diễn subword của MiniLM.
  2. **Case #12 (`natural_spam`, nhãn thật = spam):** *"hi there, sorry to bother you, my friend gave me your number, I'm new in town..."*.
     - *Cơ chế sai:* Không chứa bất kỳ từ khóa quảng cáo/trúng thưởng nào, câu chữ lịch sự như tin nhắn cá nhân nên cả BoW lẫn Embedding đều xếp vào vùng `ham`.

## 5. Thử thách cá nhân

- **Đề #6 (SEED = 3251):** *LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?*
- **Cách làm:** Xây dựng hàm `build_classification_prompt_json(text)` yêu cầu trả về đúng cấu trúc JSON 2 trường `label` và `reason`; viết `parse_json_label(response)` dùng `json.loads` kết hợp fallback Regex `r'\{.*?\}'` khi LLM bọc markdown.
- **Kết quả:**
  - Bản 1 từ: ~35.6 token input + ~2 token output → Chi phí **~$0.0046 / 1k tin**.
  - Bản JSON: ~81.3 token input + ~13.3 token output → Chi phí **~$0.0148 / 1k tin** (tăng gấp **~3.2–3.7 lần**; nếu tính riêng phần output đắt gấp 5 lần input thì chi phí sinh chữ tăng mạnh).
- **Giải thích:** Không đáng nếu chạy tự động cho toàn bộ 10 triệu SMS/ngày vì làm đội chi phí và tăng độ trễ gấp nhiều lần; nhưng rất đáng nếu chỉ áp dụng ở tầng kiểm duyệt/giải quyết khiếu nại cần lý do minh bạch (`reason`).

## 6. Phương án cho sản phẩm thật

- **Lựa chọn:** Chọn phương án **`3b. Encoder (MiniLM) + LogReg`** làm bộ lọc chính, kết hợp thiết kế **Router 2 tầng** (chỉ đẩy ~1–2% tin có xác suất lưỡng lự $p \in [0.1, 0.9]$ sang LLM hoặc `2. BoW + MLP` làm tầng lọc thô siêu tốc phía trước).
- **Dẫn số liệu:**
  - `3b. Encoder + LogReg` đạt **F1 (test) = 0.965** (Precision = 0.982, Recall = 0.949), độ trễ **p95 = 39.729 ms/tin < 50 ms** (đạt SLA), chi phí API = **$0.0**, và trên bộ Stress Test đạt **F1 = 0.702** (bền vững gấp hơn 2 lần so với BoW + LogReg chỉ đạt 0.308).
  - Trong khi đó `3a. Encoder + kNN` có p95 = **84.249 ms** (vi phạm SLA 50 ms), còn LLM có độ trễ hàng trăm ms và tốn > $1.000/tháng cho 300 triệu tin.
- **Đánh đổi chấp nhận:** Chấp nhận độ trễ p95 ~39.7 ms (cao hơn mức 3.4 ms của `BoW + MLP`) và thời gian encode ban đầu (~36.6s) để đổi lấy khả năng hiểu ngữ nghĩa, chống lại các tin spam ngụy trang từ vựng ngoài thực tế.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4: LLM | Cách viết hàm phân loại LLM batch và ước tính chi phí token | Gợi ý vòng lặp gọi `call_llm` qua `retry_with_backoff`, parse nhãn và công thức tính USD theo `PRICE_PER_1M` | Chạy kiểm tra tự động TODO 4.1 + 4.4 bằng API giả lập đạt tích xanh ✅ |
| Phần 5: Thử thách cá nhân | Cách sửa prompt trả về JSON `{"label", "reason"}` và so sánh chi phí | Viết schema JSON, hàm `parse_json_label` và tính toán chênh lệch chi phí | Tự chạy code thử thách Đề #6 trong notebook, kiểm chứng chi phí bản JSON tăng > 3 lần |

**Điều đã học hôm nay:**
Trước buổi lab, em nghĩ mô hình càng phức tạp (thêm nhiều tầng MLP hoặc dùng LLM) thì F1 sẽ luôn cao hơn. Qua thực nghiệm với số liệu của chính mình, em hiểu ra rằng khi đã có biểu diễn tốt từ pre-trained Embedding (MiniLM), một bộ phân loại tuyến tính đơn giản (`LogReg`, F1 = 0.965) lại hiệu quả và ít overfit hơn cả `MLP head` (F1 = 0.952), đồng thời trong bài toán sản phẩm thực tế cần cân bằng chặt chẽ giữa chất lượng (F1), độ trễ (SLA p95 < 50 ms) và chi phí vận hành.

