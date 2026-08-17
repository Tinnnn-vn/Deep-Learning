# Direct Preference Optimization (DPO) — Dạy mô hình nhận biết câu trả lời nào tốt hơn

## Mục lục

1. [DPO đang giải quyết vấn đề gì?](#1-dpo-đang-giải-quyết-vấn-đề-gì)
2. [Ba kiểu huấn luyện: Pre-training, SFT và DPO](#2-ba-kiểu-huấn-luyện-pre-training-sft-và-dpo)
3. [Dữ liệu preference trông như thế nào?](#3-dữ-liệu-preference-trông-như-thế-nào)
4. [Từ “A tốt hơn B” đến xác suất](#4-từ-a-tốt-hơn-b-đến-xác-suất)
5. [Từ xác suất đến loss](#5-từ-xác-suất-đến-loss)
6. [LLM tính điểm cho một câu như thế nào?](#6-llm-tính-điểm-cho-một-câu-như-thế-nào)
7. [Vì sao không dùng log-probability thô?](#7-vì-sao-không-dùng-log-probability-thô)
8. [Reference model — chiếc thước đo cố định](#8-reference-model--chiếc-thước-đo-cố-định)
9. [Lắp ráp công thức DPO](#9-lắp-ráp-công-thức-dpo)
10. [Ví dụ tính tay từ đầu đến cuối](#10-ví-dụ-tính-tay-từ-đầu-đến-cuối)
11. [Tensor thay đổi kích thước ra sao?](#11-tensor-thay-đổi-kích-thước-ra-sao)
12. [Cài đặt DPO bằng PyTorch](#12-cài-đặt-dpo-bằng-pytorch)
13. [DPO khác SFT và RLHF/PPO thế nào?](#13-dpo-khác-sft-và-rlhfppo-thế-nào)
14. [Những điều DPO không tự động giải quyết](#14-những-điều-dpo-không-tự-động-giải-quyết)
15. [Bài tập tự luyện](#15-bài-tập-tự-luyện)
16. [Tóm tắt một trang](#16-tóm-tắt-một-trang)

---

## 1. DPO đang giải quyết vấn đề gì?

Một mô hình ngôn ngữ sau khi pre-training rất giỏi đoán token tiếp theo, nhưng chưa chắc đã là một trợ lý tốt.

Ví dụ, ta nhập:

```text
Nguyên nhân chính tạo ra các mùa trên Trái Đất là gì?
```

Mô hình gốc có thể viết tiếp:

```text
A. Khoảng cách đến Mặt Trời
B. Độ nghiêng của trục Trái Đất
C. Tốc độ tự quay
D. Dòng hải lưu
```

Nó không cố tình né câu hỏi. Nó chỉ đoán rằng sau một câu hỏi kiểu này thường xuất hiện các lựa chọn trắc nghiệm.

Nói ngắn gọn:

```text
Pre-training dạy mô hình viết tiếp văn bản hợp lý.
Nó chưa trực tiếp dạy mô hình phục vụ đúng ý người dùng.
```

Để tạo một trợ lý hữu ích, ta cần dạy thêm cho mô hình những điều như:

- trả lời đúng câu hỏi;
- trình bày rõ ràng;
- không bịa đặt khi thiếu dữ kiện;
- từ chối yêu cầu nguy hiểm;
- ưu tiên câu trả lời hữu ích hơn câu trả lời chung chung.

Quá trình khiến hành vi của AI gần với mong muốn của con người thường được gọi là **alignment**.

---

## 2. Ba kiểu huấn luyện: Pre-training, SFT và DPO

Hãy tưởng tượng ta đang đào tạo một học sinh.

| Giai đoạn | Học sinh được làm gì? | Trong LLM |
|---|---|---|
| Pre-training | Đọc một thư viện khổng lồ và đoán phần tiếp theo | Học dự đoán token tiếp theo từ lượng văn bản lớn |
| SFT | Xem câu hỏi cùng bài giải mẫu rồi bắt chước | Học từ cặp `prompt → ideal_response` |
| DPO | Nhìn hai bài giải và học vì sao một bài tốt hơn | Học từ cặp `chosen > rejected` |

### 2.1 SFT làm tốt điều gì?

**Supervised Fine-Tuning (SFT)** dùng dữ liệu dạng:

```text
Câu hỏi → Câu trả lời mẫu tốt
```

Ví dụ:

```text
Prompt:
Vì sao bầu trời có màu xanh?

Ideal response:
Ánh sáng xanh bị các phân tử trong khí quyển tán xạ mạnh hơn phần lớn
các màu có bước sóng dài, nên mắt ta thấy bầu trời chủ yếu có màu xanh.
```

SFT giúp mô hình học cách trả lời như một trợ lý. Tuy nhiên, trong thực tế thường không chỉ có đúng một câu trả lời chấp nhận được.

### 2.2 Vì sao cần thêm dữ liệu so sánh?

Xét hai câu:

```text
A: DPO là một thuật toán dành cho AI.

B: DPO tinh chỉnh mô hình ngôn ngữ bằng các cặp câu trả lời,
   trong đó con người cho biết câu nào tốt hơn.
```

Cả hai không hoàn toàn sai, nhưng B rõ ràng và hữu ích hơn A.

Viết một câu trả lời hoàn hảo từ đầu khá khó. Chọn câu tốt hơn trong hai câu thường dễ hơn nhiều. DPO khai thác chính loại tín hiệu này.

> **SFT dạy mô hình bắt chước câu trả lời tốt. DPO dạy mô hình phân biệt câu nào được ưa thích hơn.**

---

## 3. Dữ liệu preference trông như thế nào?

Mỗi mẫu DPO thường có ba phần:

| Ký hiệu | Tên thường dùng | Ý nghĩa |
|---|---|---|
| $$\(x\)$$ | `prompt` | Câu hỏi hoặc yêu cầu |
| $$\(y_w\)$$ | `chosen`, `winner` | Câu trả lời được đánh giá tốt hơn |
| $$\(y_l\)$$ | `rejected`, `loser` | Câu trả lời kém hơn |

Ví dụ ở dạng Python:

```python
sample = {
    "prompt": "DPO là gì?",
    "chosen": (
        "DPO là phương pháp tinh chỉnh LLM từ dữ liệu so sánh, "
        "trong đó câu trả lời được chọn cần được ưu tiên hơn câu bị loại."
    ),
    "rejected": "DPO là một thuật toán AI."
}
```

Dữ liệu chỉ nói:

```text
Với prompt x, con người thích y_w hơn y_l.
```

Nó không khẳng định `chosen` hoàn hảo, cũng không khẳng định `rejected` luôn hoàn toàn sai. Nó chỉ cung cấp một **quan hệ so sánh**.

---

## 4. Từ “A tốt hơn B” đến xác suất

Máy tính không thể backpropagation trực tiếp qua câu nhận xét “A tốt hơn B”. Ta phải chuyển nhận xét này thành số.

### 4.1 Giả sử mỗi câu có một điểm chất lượng ẩn

Gọi:

- $$\(r_w\)$$: điểm của câu `chosen`;
- $$\(r_l\$$: điểm của câu `rejected`.

Nếu `chosen` tốt hơn, ta mong muốn:

$$
r_w > r_l
$$

Ví dụ:

$$
\[
r_w=2.5,\qquad r_l=0.8
\]
$$

Khoảng cách giữa hai câu là:

$$
\[
\Delta r=r_w-r_l=2.5-0.8=1.7
\]
$$

Điều quan trọng là **khoảng cách**, không phải điểm tuyệt đối. Cặp điểm \((10,8)\) và \((4,2)\) đều cho khoảng cách bằng 2.

### 4.2 Sigmoid biến khoảng cách thành xác suất

Hàm sigmoid:

$$
\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]
$$

Nhiệm vụ của nó rất đơn giản: nhận một số bất kỳ và ép kết quả vào khoảng từ 0 đến 1.

| $$\(z=r_w-r_l\)$$ | $$\(\sigma(z)\)$$ gần bằng | Cách hiểu |
|---:|---:|---|
| 5 | 0.993 | Gần như chắc chắn `chosen` thắng |
| 2 | 0.881 | Khá tin rằng `chosen` thắng |
| 0 | 0.500 | Hai câu ngang nhau |
| -2 | 0.119 | Mô hình đang nghiêng về `rejected` |
| -5 | 0.007 | Mô hình gần như chắc chắn chọn sai |

Mô hình Bradley–Terry viết xác suất `chosen` được thích hơn như sau:

$$
\[
P(y_w \succ y_l)=\sigma(r_w-r_l)
\]
$$

Với khoảng cách 1.7:

$$
\[
P(y_w \succ y_l)=\sigma(1.7)\approx0.845
\]
$$

Tức là hệ thống dự đoán xác suất con người chọn `chosen` khoảng **84.5%**.

---

## 5. Từ xác suất đến loss

Xác suất mới chỉ là dự đoán. Để huấn luyện, ta cần một con số cho biết mô hình đang làm chưa tốt đến mức nào. Con số đó là **loss**.

Vì nhãn cho biết `chosen` phải thắng, ta dùng:

$$
\[
\mathcal{L}=-\log P(y_w \succ y_l)
\]
$$

Thay công thức Bradley–Terry vào:

$$
\[
\mathcal{L}=-\log\sigma(r_w-r_l)
\]
$$

### 5.1 Trực giác của $$\(-\log(p)\)$$

| Xác suất `chosen` thắng | Loss $$\(-\log(p)\)$$ | Nhận xét |
|---:|---:|---|
| 0.99 | 0.010 | Dự đoán rất tốt |
| 0.90 | 0.105 | Dự đoán tốt |
| 0.50 | 0.693 | Chưa phân biệt được |
| 0.10 | 2.303 | Dự đoán sai |
| 0.01 | 4.605 | Sai và còn rất tự tin |

Ta muốn loss nhỏ dần. Vì vậy, quá trình huấn luyện sẽ cố làm cho:

$$
\[
r_w-r_l \quad \text{ngày càng lớn}
\]
$$

Một mốc rất dễ nhớ:

> Khi hai câu được chấm ngang nhau, hiệu điểm bằng 0, sigmoid bằng 0.5 và loss bằng khoảng **0.693**.

---

## 6. LLM tính điểm cho một câu như thế nào?

Ta đã có “vỏ ngoài” của bài toán:

$$\[
\mathcal{L}=-\log\sigma(r_w-r_l)
\]$$


Nhưng $$\(r_w\)$$ và $$\(r_l\)$$ lấy từ đâu?

### 6.1 LLM tạo câu từng token một

Giả sử câu trả lời có ba token:

```text
Paris | rất | đẹp
```

Mô hình tính lần lượt:

$$
\[
P(\text{Paris}\mid x)
\]
$$

$$
\[
P(\text{rất}\mid x,\text{Paris})
\]
$$

$$
\[
P(\text{đẹp}\mid x,\text{Paris rất})
\]
$$

Xác suất của cả câu là tích:

$$
\pi(y \mid x) = \prod_{t=1}^{T} \pi(y_t \mid x, y_{ < t })
$$

Ví dụ các xác suất lần lượt là 0.6, 0.5 và 0.4:

$$
P(\text{cả câu}) = 0.6 \times 0.5 \times 0.4 = 0.12
$$

### 6.2 Vì sao phải dùng log-probability?

Một câu dài yêu cầu nhân rất nhiều số nhỏ hơn 1, nên kết quả có thể nhỏ đến mức máy tính khó biểu diễn chính xác.

Logarit có quy tắc:

$$
\log(a \times b) = \log a + \log b
$$

Nhờ đó:

$$
\log \pi(y \mid x) = \sum_{t=1}^{T} \log \pi(y_t \mid x, y_{\lt t})
$$

Ta đổi một chuỗi phép nhân thành một chuỗi phép cộng ổn định hơn.

### 6.3 Vì sao log-probability thường âm?

Xác suất nằm trong đoạn từ 0 đến 1, nên logarit của nó không dương.

| Xác suất | Log tự nhiên gần bằng |
|---:|---:|
| 1.0 | 0 |
| 0.9 | -0.105 |
| 0.5 | -0.693 |
| 0.1 | -2.303 |

Do đó:

```text
-1 lớn hơn -5
→ câu có điểm -1 được mô hình xem là có khả năng cao hơn câu có điểm -5.
```

---

## 7. Vì sao không dùng log-probability thô?

Một ý tưởng ban đầu là đặt:

$$
r(x, y) = \log \pi_\theta(y \mid x)
$$

Tức là câu nào model hiện tại cho xác suất cao hơn thì được xem là tốt hơn. Ý tưởng này chưa đủ an toàn.

### 7.1 Thiên lệch theo độ dài

Tổng log-probability là tổng các số âm. Thêm token thường làm tổng điểm âm hơn.

Giả sử mỗi token có log-probability `-0.105`:

| Câu trả lời | Số token | Tổng log-probability |
|---|---:|---:|
| “Một thuật toán” | 2 | -0.210 |
| “Một thuật toán giúp căn chỉnh mô hình ngôn ngữ” | 7 | -0.735 |

Vì $$\(-0.210>-0.735\)$$, điểm thô nghiêng về câu ngắn dù câu dài có thể hữu ích hơn.

### 7.2 Xác suất cao không đồng nghĩa với chất lượng cao

Những từ quen thuộc như `“là”, “một”, “và”.` thường dễ dự đoán. Một câu chung chung có thể có xác suất cao nhưng ít thông tin:

```text
Đó là một câu hỏi rất thú vị và đây là một vấn đề đáng quan tâm.
```

Trong khi câu chứa dữ kiện cụ thể đôi khi khó đoán hơn:

```text
Hai rủi ro đáng chú ý là thiên lệch thuật toán và việc làm bị tự động hóa.
```

Vì vậy, ta không nên đồng nhất:

```text
“mô hình dễ sinh ra câu này” = “con người đánh giá câu này tốt”.
```

---

## 8. Reference model — chiếc thước đo cố định

DPO thường bắt đầu từ một model đã qua SFT rồi tạo ra hai vai trò:

| Model | Ký hiệu | Có cập nhật không? | Vai trò |
|---|---|---:|---|
| Policy model | $$\(\pi_\theta\)$$ | Có | Model đang học |
| Reference model | $$\(\pi_{\text{ref}}\)$$ | Không | Mốc ban đầu để so sánh |

Ban đầu, chúng thường là hai bản giống nhau. Trong lúc học DPO:

- policy model thay đổi;
- reference model được đóng băng;
- cùng một câu trả lời được chấm bởi cả hai model.

DPO không chỉ hỏi:

```text
Policy thích câu này đến mức nào?
```

Nó hỏi câu chính xác hơn:

```text
So với model ban đầu, policy đã tăng hay giảm mức ưu tiên cho câu này bao nhiêu?
```

### 8.1 Điểm thay đổi tương đối

Với một câu trả lời $$\(y\)$$, đặt:

$$
\Delta(y) = \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)
$$

Đọc bằng lời:

> Điểm hiện tại trừ điểm ở mốc ban đầu.

Ví dụ:

| Câu | Policy | Reference | $$\(\Delta\)$$ |
|---|---:|---:|---:|
| Chosen | -2.0 | -3.0 | +1.0 |
| Rejected | -2.5 | -2.0 | -0.5 |

Ý nghĩa:

- `chosen`: policy thích hơn trước;
- `rejected`: policy thích ít hơn trước.

Đó chính là hướng ta muốn.

### 8.2 Nhìn dưới dạng tỉ lệ

Vì:

$$
\log a - \log b = \log \frac{a}{b}
$$

nên:

$$
\Delta(y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

Nếu tạm bỏ logarit:

| Tỉ lệ policy/reference | Ý nghĩa |
|---:|---|
| Lớn hơn 1 | Policy tăng xác suất cho câu này |
| Bằng 1 | Không thay đổi |
| Nhỏ hơn 1 | Policy giảm xác suất cho câu này |

### 8.3 Một lưu ý quan trọng

Reference model giúp so sánh **mức thay đổi tương đối** và giữ policy gắn với điểm xuất phát SFT. Nó thường làm nhiều thiên lệch chung giữa hai model được triệt tiêu bớt.

Tuy nhiên, không nên nói reference model “xóa hoàn toàn” thiên lệch độ dài hay giải quyết mọi vấn đề. Chất lượng còn phụ thuộc vào dữ liệu preference, cách tokenize, cách cộng/chuẩn hóa log-probability, siêu tham số và quy trình đánh giá.

---

## 9. Lắp ráp công thức DPO

Đây là phần quan trọng nhất. Ta sẽ không nhảy thẳng vào công thức dài.

### Bước 1: Policy thay đổi thế nào với `chosen`?

$$
\Delta_w = \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)
$$

### Bước 2: Policy thay đổi thế nào với `rejected`?

$$
\Delta_l = \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)
$$

### Bước 3: So sánh hai thay đổi

$$
m = \Delta_w - \Delta_l
$$

Ta gọi $$\(m\)$$ là **preference margin**.

| Giá trị của $$\(m\)$$ | Ý nghĩa |
|---:|---|
| $$\(m>0\)$$ | Policy thay đổi theo hướng ưu tiên `chosen` |
| $$\(m=0\)$$ | Policy chưa tạo ra khác biệt tương đối |
| $$\(m<0\)$$ | Policy đang đi ngược dữ liệu preference |

### Bước 4: Đưa margin qua sigmoid và loss

Ta nhân margin với $$\(\beta\)$$, rồi dùng sigmoid:

$$
p = \sigma(\beta m)
$$

Cuối cùng:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta m)
$$

Thay $$\(m=\Delta_w-\Delta_l\)$$:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta (\Delta_w - \Delta_l) \right)
$$

Khai triển đầy đủ:

$$
\boxed{
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \left[ \left( \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x) \right) - \left( \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x) \right) \right] \right)
}
$$

Nếu huấn luyện trên cả dataset $$\(\mathcal D\)$$, ta lấy trung bình trên các mẫu:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta (\Delta_w - \Delta_l) \right) \right]
$$

### 9.1 $$\(\beta\)$$ là gì?

$$\(\beta\)$$ điều chỉnh thang đo của preference margin và xuất hiện từ cách DPO liên hệ bài toán preference với mục tiêu có ràng buộc KL so với reference model.

Điều nên nhớ:

- `beta` không phải learning rate;
- learning rate quyết định optimizer cập nhật trọng số mỗi bước lớn đến đâu;
- `beta` quyết định margin được đưa vào loss với thang đo nào;
- trong cách suy ra từ RL có KL, `beta` gắn với mức phạt rời xa reference;
- ảnh hưởng thực tế còn tương tác với optimizer, dữ liệu và cách cài đặt, nên phải kiểm chứng bằng validation.

Một giá trị khởi đầu thường gặp là `0.1`, nhưng không có con số tốt nhất cho mọi bài toán.

---

## 10. Ví dụ tính tay từ đầu đến cuối

Giả sử ta có:

```text
Prompt:   Thủ đô của Pháp là gì?
Chosen:   Paris.
Rejected: Lyon.
Beta:     0.1
```

Bốn điểm log-probability:

| Câu | Policy $$\(\pi_\theta\)$$ | Reference $$\(\pi_{\text{ref}}\)$$ |
|---|---:|---:|
| Chosen | -10 | -12 |
| Rejected | -15 | -14 |

### Bước 1: Tính thay đổi của `chosen`

$$
\Delta_w = -10 - (-12) = 2
$$

Policy đã tăng mức ưu tiên cho `chosen` so với reference.

### Bước 2: Tính thay đổi của `rejected`

$$
\Delta_l = -15 - (-14) = -1
$$

Policy đã giảm mức ưu tiên cho `rejected`.

### Bước 3: Tính preference margin

$$
m = \Delta_w - \Delta_l = 2 - (-1) = 3
$$

### Bước 4: Nhân với beta

$$
z = \beta m = 0.1 \times 3 = 0.3
$$

### Bước 5: Đổi thành xác suất

$$
p = \sigma(0.3) \approx 0.574
$$

### Bước 6: Tính loss

$$
\mathcal{L} = -\log(0.574) \approx 0.555
$$

Loss vẫn lớn hơn 0, nên optimizer còn tiếp tục điều chỉnh policy để margin tăng.

### Trường hợp lúc mới bắt đầu

Nếu policy và reference giống hệt nhau:

$$
\Delta_w = 0, \qquad \Delta_l = 0
$$

Do đó:

$$
m = 0, \quad p = 0.5, \quad \mathcal{L} \approx 0.693
$$

Điều này hợp lý: trước khi học preference, policy chưa nghiêng về phía nào so với chính mốc ban đầu của nó.

---

## 11. Tensor thay đổi kích thước ra sao?

Đây là phần rất dễ gây rối khi chuyển toán thành PyTorch.

Ký hiệu:

- $$\(B\)$$: batch size;
- $$\(P\)$$: số token của prompt;
- $$\(R\)$$: số token của response;
- $$\(V\)$$: kích thước vocabulary.

| Tensor | Shape | Chứa gì? |
|---|---|---|
| `prompt_tokens` | `[B, P]` | Token ID của prompt |
| `response_tokens` | `[B, R]` | Token ID của câu trả lời |
| `input_ids` | `[B, P + R]` | Prompt ghép response |
| `logits` | `[B, P + R, V]` | Điểm dự đoán cho mọi token trong vocabulary |
| `response_logits` | `[B, R, V]` | Logits tại các vị trí dự đoán response |
| `response_tokens.unsqueeze(-1)` | `[B, R, 1]` | Token đích, thêm trục để dùng `gather` |
| `token_log_probs` | `[B, R]` | Log-probability đúng của từng token response |
| `sequence_log_probs` | `[B]` | Tổng điểm của mỗi response trong batch |

### 11.1 Vì sao phải dịch logits một vị trí?

Với chuỗi:

```text
[A, B, C, D]
```

logits tại vị trí:

```text
A dự đoán B
B dự đoán C
C dự đoán D
D dự đoán token tiếp theo
```

Giả sử prompt dài \(P\) token. Token response đầu tiên được dự đoán từ **token prompt cuối cùng**, tức vị trí `P - 1`.

Vì vậy ta lấy:

```python
response_logits = logits[:, prompt_len - 1 : -1, :]
```

- bắt đầu tại `prompt_len - 1` để lấy dự đoán cho token response đầu tiên;
- dừng trước logits cuối vì logits cuối dự đoán token nằm sau response;
- kết quả có đúng `R` vị trí.

### 11.2 `gather` làm gì?

Tại mỗi vị trí, `log_probs` chứa điểm của **toàn bộ vocabulary**. Nhưng câu thật chỉ dùng một token.

`gather` giống như yêu cầu:

```text
Ở vị trí 1, lấy điểm của token ID 5.
Ở vị trí 2, lấy điểm của token ID 6.
...
```

Nó không chọn token mới; nó chỉ lấy điểm của token đã có trong response.

---

## 12. Cài đặt DPO bằng PyTorch

Đoạn mã dưới đây trình bày phần lõi để học. Nó giả sử các chuỗi trong batch đã được padding phù hợp và `response_mask` cho biết token nào là dữ liệu thật.

### 12.1 Tính log-probability của response

```python
import torch
import torch.nn.functional as F


def get_sequence_log_probs(
    model,
    prompt_tokens,
    response_tokens,
    response_mask=None,
):
    """
    Tính tổng log-probability của response khi đã biết prompt.

    Shapes:
        prompt_tokens:   [B, P]
        response_tokens: [B, R]
        response_mask:   [B, R], 1 = token thật, 0 = padding
        return:          [B]
    """

    # [B, P] + [B, R] -> [B, P + R]
    input_ids = torch.cat([prompt_tokens, response_tokens], dim=1)

    # [B, P + R] -> [B, P + R, V]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    prompt_len = prompt_tokens.size(1)

    # Token response đầu được dự đoán tại vị trí prompt_len - 1.
    # Bỏ logits cuối vì nó dự đoán token sau response.
    # Shape: [B, R, V]
    response_logits = logits[:, prompt_len - 1 : -1, :]

    # Đổi logits thành log-probabilities trên trục vocabulary.
    # Shape vẫn là [B, R, V]
    all_log_probs = F.log_softmax(response_logits, dim=-1)

    # [B, R] -> [B, R, 1], lấy đúng token đích tại mỗi vị trí.
    # Sau squeeze: [B, R]
    token_log_probs = torch.gather(
        all_log_probs,
        dim=2,
        index=response_tokens.unsqueeze(-1),
    ).squeeze(-1)

    if response_mask is not None:
        token_log_probs = token_log_probs * response_mask

    # Cộng theo chiều token: [B, R] -> [B]
    return token_log_probs.sum(dim=1)
```

### 12.2 Một bước huấn luyện DPO

```python
def dpo_training_step(
    policy_model,
    reference_model,
    optimizer,
    batch,
    beta=0.1,
):
    """Thực hiện một bước cập nhật policy model bằng DPO."""

    policy_model.train()
    reference_model.eval()
    optimizer.zero_grad()

    # 1. Policy chấm chosen và rejected.
    # Hai nhánh này phải giữ computation graph để backward.
    policy_chosen = get_sequence_log_probs(
        policy_model,
        batch["prompt"],
        batch["chosen"],
        batch.get("chosen_mask"),
    )

    policy_rejected = get_sequence_log_probs(
        policy_model,
        batch["prompt"],
        batch["rejected"],
        batch.get("rejected_mask"),
    )

    # 2. Reference chỉ làm mốc, không cần gradient.
    with torch.no_grad():
        ref_chosen = get_sequence_log_probs(
            reference_model,
            batch["prompt"],
            batch["chosen"],
            batch.get("chosen_mask"),
        )

        ref_rejected = get_sequence_log_probs(
            reference_model,
            batch["prompt"],
            batch["rejected"],
            batch.get("rejected_mask"),
        )

    # 3. Mức thay đổi tương đối so với reference.
    chosen_change = policy_chosen - ref_chosen
    rejected_change = policy_rejected - ref_rejected

    # 4. Preference margin và DPO loss.
    preference_margin = chosen_change - rejected_change
    loss = -F.logsigmoid(beta * preference_margin).mean()

    # 5. Chỉ policy model được cập nhật.
    loss.backward()
    optimizer.step()

    # Metric: tỷ lệ mẫu trong batch có margin đúng hướng.
    preference_accuracy = (preference_margin > 0).float().mean()

    return {
        "loss": loss.item(),
        "preference_accuracy": preference_accuracy.item(),
        "mean_margin": preference_margin.detach().mean().item(),
    }
```

### 12.3 Đọc đoạn code như tiếng Việt

```text
policy_chosen - ref_chosen
→ policy đã thay đổi thế nào với câu được chọn?

policy_rejected - ref_rejected
→ policy đã thay đổi thế nào với câu bị loại?

chosen_change - rejected_change
→ policy có ưu tiên chosen hơn rejected theo nghĩa tương đối không?

-logsigmoid(beta * preference_margin)
→ nếu chưa ưu tiên đúng, model phải chịu mức phạt bao nhiêu?
```

### 12.4 Vì sao dùng `torch.no_grad()` cho reference?

Reference model không được cập nhật. `torch.no_grad()` giúp:

- không lưu computation graph không cần thiết;
- giảm bộ nhớ GPU;
- giảm công việc tính toán liên quan đến gradient.

Ta cũng nên đóng băng tham số của reference ngay từ đầu:

```python
for parameter in reference_model.parameters():
    parameter.requires_grad = False
```

### 12.5 Lưu ý khi chuyển code minh họa thành code thực tế

Đoạn code trên cố ý giữ cấu trúc đơn giản. Trong dự án thật, cần xử lý thêm:

- prompt trong batch có độ dài khác nhau;
- vị trí padding của prompt và response;
- `attention_mask` khi gọi model;
- chỉ tính loss trên response, không tính trên prompt hoặc padding;
- token kết thúc `EOS`;
- mixed precision (`bf16`/`fp16`), gradient accumulation và distributed training;
- kiểm tra cách tokenizer padding bên trái hay bên phải;
- tránh giữ hai bản model đầy đủ nếu GPU không đủ VRAM.

Thư viện huấn luyện chuyên dụng thường đã giải quyết nhiều chi tiết này. Tuy nhiên, hiểu bản lõi ở trên giúp ta không sử dụng framework như một “hộp đen”.

---

## 13. DPO khác SFT và RLHF/PPO thế nào?

| Phương pháp | Dữ liệu chính | Học điều gì? | Thành phần đáng chú ý |
|---|---|---|---|
| SFT | `prompt → response tốt` | Bắt chước câu trả lời mẫu | Một policy model |
| RLHF/PPO cổ điển | Dữ liệu preference + rollout | Tối đa reward trong vòng lặp RL | Reward model, policy, reference, thuật toán RL |
| DPO | `prompt, chosen, rejected` | Tăng ưu tiên tương đối cho chosen | Policy + reference, loss phân loại trực tiếp |

DPO hấp dẫn vì không cần huấn luyện một reward model riêng rồi chạy vòng lặp PPO phức tạp. Nhưng “đơn giản hơn” không có nghĩa là miễn phí:

- vẫn cần dữ liệu preference tốt;
- thường phải chạy cả policy và reference khi huấn luyện;
- vẫn phải chọn siêu tham số và đánh giá cẩn thận;
- dữ liệu thiên lệch sẽ dạy model thiên lệch theo.

---

## 14. Những điều DPO không tự động giải quyết

DPO là một mục tiêu huấn luyện, không phải phép màu.

### 14.1 `Chosen` kém thì model vẫn học điều kém

Nếu người gán nhãn ưu tiên câu dài dòng, sai sự thật hoặc thiên lệch, model có thể học đúng những sở thích đó.

### 14.2 Preference có thể mơ hồ

Hai người có thể không đồng ý về:

- mức độ chi tiết;
- phong cách;
- tính hài hước;
- cách cân bằng giữa an toàn và hữu ích.

Vì vậy phải có hướng dẫn gán nhãn rõ ràng và đo mức độ đồng thuận.

### 14.3 Model có thể overfit phong cách

Nếu `chosen` luôn dài hơn hoặc luôn mở đầu bằng cùng một câu, model có thể học dấu hiệu bề mặt thay vì chất lượng thật.

### 14.4 Preference accuracy không đủ để đánh giá model

Một model có `preference_accuracy` cao trên tập huấn luyện vẫn có thể:

- trả lời sai kiến thức;
- suy luận kém trên câu hỏi mới;
- dài dòng;
- giảm chất lượng ở những năng lực trước đó.

Cần đánh giá thêm factuality, helpfulness, safety, độ đa dạng và khả năng tổng quát hóa.

---

## 15. Bài tập tự luyện

### Bài 1 — Sigmoid và loss

Cho:

$$
r_w = 1.2, \qquad r_l = 0.2
$$

Hãy tính:

1. $$\(r_w-r_l\)$$;
2. $$\(\sigma(r_w-r_l)\)$$;
3. $$\(-\log\sigma(r_w-r_l)\)$$.

<details>
<summary>Xem đáp án</summary>

$$
r_w - r_l = 1
$$

$$
\sigma(1) \approx 0.731
$$

$$
-\log(0.731) \approx 0.313
$$

</details>

### Bài 2 — Tính preference margin

Cho bảng:

| Câu | Policy | Reference |
|---|---:|---:|
| Chosen | -4.0 | -5.5 |
| Rejected | -3.0 | -3.2 |

Hãy tính $$\(\Delta_w\)$$, $$\(\Delta_l\)$$ và $$\(m\)$$. Policy đang thay đổi đúng hướng không?

<details>
<summary>Xem đáp án</summary>

$$
\Delta_w = -4 - (-5.5) = 1.5
$$

$$
\Delta_l = -3 - (-3.2) = 0.2
$$

$$
m = 1.5 - 0.2 = 1.3 > 0
$$

Policy đang thay đổi đúng hướng: nó tăng ưu tiên cho cả hai câu, nhưng tăng cho `chosen` nhiều hơn.

</details>

### Bài 3 — Kiểm tra shape

Cho:

```text
B = 4, P = 6, R = 3, V = 50_000
```

Hãy viết shape của `input_ids`, `logits`, `response_logits`, `token_log_probs` và kết quả cuối.

<details>
<summary>Xem đáp án</summary>

```text
input_ids         [4, 9]
logits            [4, 9, 50_000]
response_logits   [4, 3, 50_000]
token_log_probs   [4, 3]
sequence_logprobs [4]
```

</details>

### Bài 4 — Câu hỏi tư duy

Nếu policy tăng log-probability của `chosen` thêm 0.5 nhưng tăng log-probability của `rejected` thêm 0.8, margin sẽ tốt hơn hay xấu đi?

<details>
<summary>Xem đáp án</summary>

Margin thay đổi:

$$
0.5 - 0.8 = -0.3
$$

Nó xấu đi. DPO quan tâm đến **sự chênh lệch giữa hai mức thay đổi**, không chỉ việc xác suất `chosen` có tăng hay không.

</details>

---

## 16. Tóm tắt một trang

### Dữ liệu

```text
prompt x
chosen y_w
rejected y_l
```

### Bốn lần chấm điểm

```text
Policy chấm chosen
Policy chấm rejected
Reference chấm chosen
Reference chấm rejected
```

### Hai mức thay đổi

$$
\Delta_w = \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)
$$

$$
\Delta_l = \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)
$$

### Một preference margin

$$
m = \Delta_w - \Delta_l
$$

### Một loss

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta m)
$$

### Một câu để nhớ

> **So với model ban đầu, hãy làm cho policy tăng mức ưu tiên dành cho câu được chọn nhiều hơn mức ưu tiên dành cho câu bị loại.**

---

## Gợi ý lộ trình học tiếp

Sau khi hiểu tài liệu này, bạn có thể học theo thứ tự:

1. tự tính DPO loss bằng bốn số log-probability;
2. chạy hàm PyTorch với tensor giả;
3. học masking và padding cho batch có độ dài khác nhau;
4. fine-tune một model nhỏ với PEFT/LoRA;
5. dùng một thư viện DPO trainer;
6. so sánh chất lượng trước và sau huấn luyện trên tập validation riêng.

Nếu bạn có thể tự giải thích được ba dòng sau, bạn đã nắm được phần lõi của DPO:

```python
chosen_change = policy_chosen - ref_chosen
rejected_change = policy_rejected - ref_rejected
loss = -F.logsigmoid(beta * (chosen_change - rejected_change)).mean()
```

---

## Ghi chú

Tài liệu này tập trung vào trực giác và phần lõi của DPO. Các hệ thống huấn luyện quy mô lớn còn có nhiều quyết định kỹ thuật liên quan đến cách tạo cặp preference, lọc dữ liệu, tokenizer, chuẩn hóa độ dài, phân phối batch, hiệu quả bộ nhớ và đánh giá an toàn.
