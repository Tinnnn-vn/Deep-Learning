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

Nhờ phép nhân ma trận này, từ một ma trận $X$ ban đầu, máy tính tách thành 3 ma trận chức năng riêng biệt để chuẩn bị đi "so khớp" xem từ nào cần chú ý đến từ nào.

**Ví dụ:**

Bây giờ chúng ta sẽ thêm ví dụ để thực hành với những phép toán giúp trực quan hơn, trong ví dụ này tôi sẽ đưa ra một câu chỉ 3 từ để dễ thực hiện phép toán bằng tay hơn.

Câu: "Tôi học AI"

Ta có 3 token:

```
Token 1 = Tôi
Token 2 = học
Token 3 = AI
```

Mỗi token sẽ được biểu diễn bằng vector 2 chiều. Như vậy `số token = 3`, `số chiều mỗi vector = 2`

Do đó:
```
Q có shape: 3 × 2
K có shape: 3 × 2
V có shape: 3 × 2
```

Giả sử ta có sẵn Q, K, V và chọn các số đơn giản để thực hiện phép tính:

```
Q =
[
  [1, 0],   ← Query của "Tôi"
  [0, 1],   ← Query của "học"
  [1, 1]    ← Query của "AI"
]
```

```
K =
[
  [1, 0],   ← Key của "Tôi"
  [0, 1],   ← Key của "học"
  [1, 1]    ← Key của "AI"
]
```

```
V =
[
  [10, 0],   ← Value của "Tôi"
  [0, 10],   ← Value của "học"
  [10, 10]   ← Value của "AI"
]
```

Ở đây ta cố tình chọn: `Q = K`. Mục đích là để dễ nhìn

Từ công thức: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Chúng ta sẽ bắt đầu mổ xẻ từng phần

**B1. Tính Kᵀ 

Ma trận K ban đầu có `shape: 3 x 2`. Vì vậy, ta cần chuyển vị (chuyển hàng thành cột):
```
Kᵀ =
[
  [1, 0, 1],
  [0, 1, 1]
]
```
Lúc này, Kᵀ có `shape: 2 × 3`

**B2. Tính QKᵀ**

Mỗi token sẽ so sánh với cả 3 token. Ta tính từng ô (hàng 1 x cột 1, hàng 1 x cột 2, hàng 1 x cột 3,.v.v.)

```
QKᵀ =
[
  [1, 0],
  [0, 1],
  [1, 1]
]
×
[
  [1, 0, 1],
  [0, 1, 1]
]
```
Do đó:
```
QKᵀ =
[
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 2]
]
```

Bây giờ ta đã thu được ma trận QKᵀ. Ma trận này gọi là điểm số chú ý (attention scores).

Đọc theo hàng:
```
Hàng 1: "Tôi" chú ý đến [Tôi, học, AI]
Hàng 2: "học" chú ý đến [Tôi, học, AI]
Hàng 3: "AI" chú ý đến [Tôi, học, AI]
```

**B3. Chia cho $\sqrt{d_k}$**

Ở đây vector Key có 2 chiều: `dₖ = 2`. Vậy nên `√dₖ = √2 ≈ 1.414`

Ta chia từng phần tử trong QKᵀ cho 1.414:
```
QKᵀ / √dₖ =
[
  [1/1.414, 0/1.414, 1/1.414],
  [0/1.414, 1/1.414, 1/1.414],
  [1/1.414, 1/1.414, 2/1.414]
]
```

Xấp xỉ:
```
Scaled Scores =
[
  [0.707, 0.000, 0.707],
  [0.000, 0.707, 0.707],
  [0.707, 0.707, 1.414]
]
```

Đây vẫn là điểm chú ý, nhưng đã được làm “mềm” hơn\

**B4. Softmax từng hàng**

Bây giờ ta cần biến mỗi hàng thành xác suất. Vì mỗi hàng là một token đang hỏi: `“Tôi nên chú ý đến các token khác bao nhiêu phần trăm?”`

Công thức softmax cho một vector:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

**Tính Softmax hàng 1:**
```
e^0.707 ≈ 2.028
e^0.000 = 1
e^0.707 ≈ 2.028
```
Tổng: `2.028 + 1 + 2.028 = 5.056`

Chia cho Tổng:
```
2.028 / 5.056 ≈ 0.401
1.000 / 5.056 ≈ 0.198
2.028 / 5.056 ≈ 0.401
```
Vậy hàng 1 sau softmax sẽ là: `[0.401, 0.198, 0.401]`

Nghĩa là:
```
"Tôi" chú ý:
40.1% đến "Tôi"
19.8% đến "học"
40.1% đến "AI"
```
Giờ hàng 2 và 3 cũng thực hiện tương tự như hàng 1, ta thu được kết quả sau:

Hàng 2 sau softmax: `[0.198, 0.401, 0.401]`

Hàng 3 sau softmax: `[0.248, 0.248, 0.504]`

**Giờ ta có Attention Weights**

Sau softmax, ta có ma trận trọng số chú ý sau:

```
A =
[
  [0.401, 0.198, 0.401],
  [0.198, 0.401, 0.401],
  [0.248, 0.248, 0.504]
]
```

Đây chính là: $$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Mỗi hàng cộng lại xấp xỉ bằng 1:

```
0.401 + 0.198 + 0.401 = 1.000
0.198 + 0.401 + 0.401 = 1.000
0.248 + 0.248 + 0.504 = 1.000
```

**B5. Nhân Attention Weights với V**




















