# Error analysis

## 1. Các mẫu được phân loại đúng

| Text | True Label | Predicted Label |
|---|---|---|
| can you plz tell me the ans. bslvyl sent via fullonsms.com	 | ham | ham |
| you intrepid duo you! have a great time and see you both soon. | ham | ham |
| r u in this continent? | ham | ham |

## 2. Các mẫu bị phân loại sai

| Text | True Label | Predicted Label |
|---|---|---|
| it to 80488. your 500 free text messages are valid until 31 december 2005. | spam | ham |
| sir, i am waiting for your call, once free please call me. | ham | spam |
| free game. get rayman golf 4 free from the o2 games arcade. 1st get ur games settings. reply post, then save & activ8. press 0 key for arcade. termsapply	 | spam | ham |

## 3. Bảng phân tích từng mẫu sai

| Yếu tố | Mẫu 1: Spam → Ham | Mẫu 2: Ham → Spam | Mẫu 3: Spam → Ham |
|---|---|---|---|
| **Lexical ambiguity (Mơ hồ từ vựng)** | Có thể. Từ `free` có thể mang nghĩa “miễn phí” hoặc “rảnh”, nhưng trong câu này có nghĩa là tin nhắn miễn phí. | Có. Từ `free` mang nghĩa “rảnh”, không phải “miễn phí”. Mô hình có thể liên kết từ này với spam. | Có thể. Từ `free` mang nghĩa “miễn phí”, nhưng từ này có thể xuất hiện trong nhiều loại tin nhắn khác nhau. |
| **Negation (Phủ định)** | Không có dấu hiệu phủ định rõ ràng. | Không có dấu hiệu phủ định rõ ràng. | Không có dấu hiệu phủ định rõ ràng. |
| **Sarcasm (Mỉa mai)** | Không có dấu hiệu mỉa mai rõ ràng. | Không có dấu hiệu mỉa mai rõ ràng. | Không có dấu hiệu mỉa mai rõ ràng. |
| **Short text (Văn bản ngắn)** | Không phải nguyên nhân nổi bật vì tin nhắn có nội dung tương đối đầy đủ. | Có thể là yếu tố phụ vì tin nhắn khá ngắn, nhưng vẫn có đủ ngữ cảnh để hiểu nội dung. | Không phải nguyên nhân nổi bật vì tin nhắn có nhiều thông tin quảng cáo. |
| **Rare word (Từ hiếm)** | Có thể. `80488` là một chuỗi số đặc biệt, có thể ít xuất hiện trong dữ liệu huấn luyện. | Không có từ hiếm nổi bật rõ ràng. | Có. `Rayman`, `activ8` và `termsapply` có thể ít phổ biến hoặc là cách viết đặc biệt. |
| **Domain-specific term (Thuật ngữ chuyên ngành)** | Có. `80488` có thể là mã dịch vụ; cụm `500 free text messages` liên quan đến dịch vụ viễn thông. | Không rõ. Nội dung chủ yếu là cuộc hội thoại thông thường về việc chờ một cuộc gọi. | Có. `O2 games arcade`, `games settings` và `activ8` liên quan đến dịch vụ trò chơi di động. |
| **Data noise (Nhiễu dữ liệu)** | Không có bằng chứng rõ ràng về lỗi dữ liệu. | Không có bằng chứng rõ ràng về lỗi dữ liệu. | Có thể. `termsapply` có vẻ bị dính từ hoặc thiếu dấu cách, nhưng cũng có thể là cách viết trong dữ liệu SMS. |
| **Class imbalance (Mất cân bằng lớp)** | Có thể góp phần. Lớp ham chiếm đa số nên mô hình có thể thiên về dự đoán ham. Tuy nhiên, chưa thể khẳng định đây là nguyên nhân trực tiếp. | Có thể ảnh hưởng đến quá trình học, nhưng không giải thích trực tiếp tại sao mô hình dự đoán nhầm thành spam. | Có thể góp phần khiến mô hình bỏ sót một số tin nhắn spam vì lớp ham chiếm đa số trong tập dữ liệu. |