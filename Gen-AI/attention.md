# Attention Thật Đơn Giản

### I. Giới thiệu: Vấn đề của LLM khi không có Attention

Giả sử bạn đang đọc câu sau: `"Con báo đang đuổi theo con mồi trên đồng cỏ"`

Và câu này: `"Bản báo cáo tài chính quý này quá phức tạp."`

Cả hai câu đều có từ "báo" nhưng ở mỗi câu, bộ não của chúng ta lại tự động liên tưởng đến một ý nghĩa hoàn toàn khác nhau nhờ vào các từ xung quanh nó.

Từ "báo" trong câu đầu (động vật) hoàn toàn khác với từ "báo" trong câu thứ hai (tài liệu). Đối với máy móc, đây là một vấn đề lớn. Làm thế nào mà máy móc có thể hiểu ý nghĩa các từ lân cận?

Cơ chế Attention chính là cách mà máy tính bắt chước bộ não của con người, nó tính toán xem một từ cần phải "chú ý" đến những từ nào khác xung quanh để hiểu đúng ngữ cảnh của chính nó.

Trong các chương tiếp theo bạn sẽ hiểu chính xác cách thức hoạt động của các thành phần sau:

- Các từ nhúng (Word Embeddings): Điểm khởi đầu cố định cho mỗi từ.
- Chú ý tích vô hướng được mở rộng (Scaled Dot-Product Attention): Công thức cốt lõi cho phép tạo ngữ cảnh.
- Truy vấn (Query), Khóa (Key) và Giá trị (Value) (QKV): Ba vai trò mà một từ có thể đảm nhiệm.
- Mặt nạ nhân quả (Causal Mask): Làm thế nào để ngăn chặn mô hình gian lận bằng cách nhìn trước tương lai.
- Cơ chế chú ý đa đầu (Multi-Head Attention): Làm thế nào để mở rộng cơ chế này cho các mô hình mạnh mẽ.

<img width="609" height="331" alt="Screenshot 2026-07-01 203258" src="https://github.com/user-attachments/assets/1aeb6412-c853-41ce-a7c9-acab392da540" />

### 1. Scaled Dot-Product Attention

Công thức cốt lõi này và các đoạn mã ở những chương sau là xương sống của các mô hình như ChatGPT, Gemini,.v.v.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### II. Giải thích toán học và trực giác

**Biến ngôn ngữ thành những con số:**

Trong toán học hình học, để máy tính hiểu được một từ, người ta biến từ đó thành một vector (bạn có thể tưởng tượng nó như một mũi tên xuất phát từ gốc tọa độ $(0,0)$ trong hệ trục tọa độ).
Ví dụ bằng một không gian 2 chiều gồm trục $(X, Y)$:
- Trục $X$: Đại diện cho yếu tố "Động vật" (càng gần 1 là càng thuộc về động vật, càng gần 0 là không phải).
- Trục $Y$: Đại diện cho yếu tố "Hành động săn mồi" (càng gần 1 là càng liên quan đến săn mồi, càng gần 0 là không).

Bây giờ, chúng ta thử đặt 3 từ sau vào không gian này dưới dạng các điểm tọa độ $(X, Y)$:
- Từ "báo" (con báo) có tọa độ là $(0.9, 0.9)$ vì nó vừa là động vật, vừa săn mồi rất giỏi.
- Từ "con mồi" có tọa độ là $(0.85, 0.1)$ vì nó là động vật nhưng không phải là kẻ đi săn mồi.
- Từ "bài báo" (tờ báo giấy) có tọa độ là $(0.0, 0.0)$ vì nó không phải động vật cũng không đi săn.

Nếu bạn vẽ các mũi tên từ gốc tọa độ $(0,0)$ đến 3 điểm này, bạn sẽ thấy mũi tên của từ "báo" và từ "con mồi" tạo với nhau một góc khá nhọn (chúng chỉa về hướng tương tự nhau trên trục động vật). Trong khi đó, mũi tên của "bài báo" lại nằm xa hẳn.

Trong hình học, để biết hai mũi tên (vector) có đang chỉ về cùng một hướng hay không, người ta thường đo góc giữa hai mũi tên đó.

Khi góc giữa hai mũi tên bằng $0^\circ$ (tức là chúng hoàn toàn trùng nhau và chỉ về cùng một hướng), thì giá trị của $\cos(0^\circ)$ sẽ bằng 1. Khi giá trị $\cos$ của góc giữa hai vector tiến gần về $1$, điều đó có nghĩa là hai vector đó đang chỉ về cùng một hướng (hoặc góc giữa chúng rất nhỏ). Trong AI, chúng ta gọi đây là độ tương đồng Cosine (Cosine Similarity).

Bây giờ hãy ráp nối hình học này vào cơ chế Attention:
- Khi mô hình LLM thấy từ "báo" và từ "con mồi" có hai vector chỉ về hướng gần nhau (góc nhỏ, $\cos$ gần bằng $1$), máy tính sẽ hiểu: "À, hai từ này có sự liên quan lớn với nhau trong không gian ý nghĩa!"
- Ngược lại, vector của từ "báo" và từ "bài báo" sẽ tạo với nhau một góc lớn hơn ( $\cos$ sẽ gần bằng $0$), nghĩa là chúng ít liên quan đến nhau hơn trong ngữ cảnh này.Để máy tính tính toán được điều này một cách tự động cho hàng triệu từ cùng một lúc, các nhà khoa học không chỉ dùng một vector cho mỗi từ, mà họ chia mỗi từ thành 3 mũi tên (vector) chức năng khác nhau, gọi tắt là Q, K, và V.
- Trong máy tính, để làm việc với hàng ngàn từ cùng một lúc, người ta không tính toán từng cặp vector mà gộp tất cả các vector của các từ lại thành một bảng số lớn, gọi là Ma trận

Ba mũi tên chức năng (Q, K, V) thực chất là viết tắt của:

- Query (Q - Câu hỏi): Đại diện cho từ đang muốn "đi tìm kiếm" ngữ cảnh. 🔍
- Key (K - Từ khóa): Đại diện cho "nhãn" của tất cả các từ trong câu để từ khác đối chiếu. 🏷️
- Value (V - Giá trị): Chứa thông tin ngữ nghĩa thực sự của từ đó khi đã tìm được sự liên quan. 💎

**Ma trận Q, K, V được tạo ra như thế nào?**

**B1: Biến từ thành Vector Nhúng (Embedding Vector)**

Trước khi có $Q, K, V$, mỗi từ được chuyển thành một chuỗi số ban đầu gọi là vector nhúng ($X$).

Ví dụ câu "Con báo" có 2 từ, ta gộp lại thành ma trận dòng $X$:

- Từ "Con" $\rightarrow X_{con} = [0.1, 0.2]$
- Từ "báo" $\rightarrow X_{báo} = [0.9, 0.8]$

**B2: Nhân với các ma trận trọng số (Weight Matrices)**

Mô hình AI có sẵn 3 ma trận hệ số cố định gọi là $W_Q, W_K, W_V$ (đây là các "bộ lọc" mà mô hình đã học được trong quá trình huấn luyện). Để tạo ra $Q, K, V$ cho toàn bộ câu, máy tính chỉ cần lấy ma trận từ đầu vào $X$ nhân lần lượt với 3 ma trận trọng số này:

- $$Q = X \times W_Q$$
- $$K = X \times W_K$$
- $$V = X \times W_V$$









































