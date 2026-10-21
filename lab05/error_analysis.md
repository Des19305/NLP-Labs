# Error Analysis

## 1. Ba mẫu phân loại đúng

| Text | True label | Predicted label | Positive probability | Correct | Prediction confidence |
|---|---|---|---:|---|---:|
| please give this one a miss br br kristy swans... | negative | negative | 0.008869 | True | 0.991131 |
| this film requires a lot of patience because i... | positive | positive | 0.998241 | True | 0.998241 |
| many animation buffs consider the great forgot... | positive | positive | 0.786935 | True | 0.786935 |

## 2. Ba mẫu phân loại sai

| Text | True label | Predicted label | Positive probability | Correct | Prediction confidence |
|---|---|---|---:|---|---:|
| originally supposed to be just a part of a hug... | positive | negative | 0.098054 | False | 0.901946 |
| i started watching this because i thought it w... | negative | positive | 0.545698 | False | 0.545698 |
| the back of the dvd for this movie raves is th... | negative | positive | 0.584489 | False | 0.584489 |

## 3. Phân tích 3 mẫu bị phân loại sai

| Mẫu | Yếu tố có thể gây lỗi | Có phải do sequence modeling? |
|---|---|---|
| 1. Review phim *Battleship Potemkin* | **Rare word, domain-specific term, long-range dependency, truncation.** Review dài 715 token, chứa nhiều thuật ngữ điện ảnh và lịch sử. Với `max_length=200`, phần lớn nội dung bị cắt bỏ. | **Một phần.** Mô hình có thể mất ngữ cảnh ở phần sau do truncation, nhưng từ hiếm và thuật ngữ chuyên ngành cũng góp phần gây lỗi. |
| 2. Review phim kinh dị | **Sarcasm, ambiguous sentiment, long-range dependency.** Review sử dụng lời châm biếm để thể hiện sự không hài lòng; một số cụm từ tích cực có thể gây nhiễu. | **Một phần.** Mô hình có thể gặp khó khăn khi kết hợp ngữ cảnh để hiểu ý mỉa mai, nhưng đây cũng là vấn đề hiểu ngôn ngữ và sắc thái cảm xúc. |
| 3. Review phim Christian | **Negation, contrast, ambiguous sentiment, truncation.** Review có các cụm như "some nice crane shots" và "special effects were decent", nhưng tổng thể mang đánh giá tiêu cực. Review dài 259 token nên bị cắt khi `max_length=200`. | **Một phần.** Mô hình có thể chú ý quá mức đến từ tích cực và bỏ sót ngữ cảnh phủ định hoặc tương phản; việc cắt chuỗi cũng có thể ảnh hưởng. |