# Reflection

| Representation           | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
| ------------------------ | ------------------ | ------------ | ----------------------- |
| **TF-IDF**               | Không              | Sparse       | Không                   |
| **Co-occurrence**        | Không              | Sparse       | Không                   |
| **Word2Vec**             | Không              | Dense        | Không                   |
| **Contextual embedding** | Có                 | Dense        | Có                      |

* Bank cần contextual representation vì nó có nhiều nghĩa phụ thuộc vào ngữ cảnh. Ví dụ, “bank” có thể nghĩa là ngân hàng hoặc bờ sông. Word2Vec chỉ tạo một vector cố định cho “bank”, trong khi contextual embedding tạo vector khác nhau dựa trên các từ xung quanh, giúp phân biệt các nghĩa khác nhau của từ