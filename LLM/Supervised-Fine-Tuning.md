# Supervised Fine-Tuning (SFT): Dạy LLM biết làm theo hướng dẫn

## 1. SFT giải quyết vấn đề gì?

Một base model vừa hoàn thành pre-training chưa thật sự là trợ lý. Nó giống một học sinh đã đọc rất nhiều sách nhưng chưa được dạy cách trả lời người dùng.

Nhiệm vụ gốc của nó chỉ là:

> Nhìn các token phía trước và đoán token tiếp theo.

Ví dụ, khi nhận được:

```text
Thủ đô của Việt Nam là gì?
```

base model có thể nối tiếp thành:

```text
A. Hà Nội
B. Đà Nẵng
C. Thành phố Hồ Chí Minh
```

Nó không cố tình né câu hỏi. Trong dữ liệu Internet, câu hỏi kiểu này thường xuất hiện trong đề trắc nghiệm, nên việc tạo thêm các đáp án là một cách nối văn bản hợp lý.

**Supervised Fine-Tuning (SFT)** giúp thay đổi hành vi đó. Ta đưa cho mô hình nhiều ví dụ gồm:

```text
(yêu cầu của người dùng, câu trả lời mẫu)
```

Ví dụ:

```text
Yêu cầu: Giải thích lực hấp dẫn cho học sinh lớp 6.
Trả lời: Lực hấp dẫn là lực kéo mọi vật về phía Trái Đất...
```

Mô hình học cách bắt chước những câu trả lời tốt. Có thể hình dung:

- **Pre-training:** đọc cả thư viện để học ngôn ngữ và kiến thức.
- **SFT:** học với một bộ bài mẫu để biết cách trả lời đúng vai trò.

```mermaid
flowchart TD
    A["Base model: nối tiếp văn bản"] --> B["Dữ liệu: yêu cầu + trả lời"]
    B --> C["Supervised Fine-Tuning"]
    C --> D["Mô hình làm theo hướng dẫn"]
```

---

## 2. SFT hoạt động như thế nào?

LLM không nhận riêng một ô “câu hỏi” và một ô “câu trả lời”. Nó nhận một chuỗi token liên tục. Vì vậy, ta ghép hai phần bằng **chat template**:

```text
<|user|> Thủ đô của Việt Nam là gì? <|end|>
<|assistant|> Thủ đô của Việt Nam là Hà Nội. <|end|>
```

Các token đặc biệt có vai trò như biển báo:

| Token | Ý nghĩa |
|---|---|
| `<|user|>` | Người dùng bắt đầu nói |
| `<|assistant|>` | Đến lượt trợ lý trả lời |
| `<|end|>` | Kết thúc một lượt nói |

Quy trình tổng quát:

1. Chuẩn bị nhiều cặp `prompt` và `response` chất lượng cao.
2. Ghép chúng theo chat template của mô hình.
3. Tokenizer đổi văn bản thành các số nguyên gọi là `input_ids`.
4. Mô hình dự đoán token tiếp theo ở từng vị trí.
5. Chỉ tính lỗi trên phần trả lời của trợ lý.
6. Backpropagation cập nhật trọng số để lần sau mô hình trả lời gần mẫu hơn.

Điểm số 5 là phần quan trọng nhất của SFT.

---
