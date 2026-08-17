# LoRA từ con số 0: Fine-tune mô hình lớn bằng những ma trận nhỏ

> Mục tiêu của bài này là hiểu **LoRA làm gì, vì sao tiết kiệm và cách tự viết LoRA**, thay vì chỉ sao chép mã từ thư viện.

---

## Mục lục

1. [Bài toán mà LoRA giải quyết](#1-bài-toán-mà-lora-giải-quyết)
2. [Ôn lại lớp tuyến tính `nn.Linear`](#2-ôn-lại-lớp-tuyến-tính-nnlinear)
3. [Ý tưởng cốt lõi của LoRA](#3-ý-tưởng-cốt-lõi-của-lora)
4. [Theo dõi hình dạng tensor](#4-theo-dõi-hình-dạng-tensor)
5. [Một ví dụ số nhỏ](#5-một-ví-dụ-số-nhỏ)
6. [LoRA tiết kiệm đến mức nào?](#6-lora-tiết-kiệm-đến-mức-nào)
7. [Tự xây dựng `LoRALinear` bằng PyTorch](#7-tự-xây-dựng-loralinear-bằng-pytorch)
8. [Gắn LoRA vào một mô hình](#8-gắn-lora-vào-một-mô-hình)
9. [Chương trình hoàn chỉnh có thể chạy](#9-chương-trình-hoàn-chỉnh-có-thể-chạy)
10. [LoRA nằm ở đâu trong Transformer?](#10-lora-nằm-ở-đâu-trong-transformer)
11. [Các siêu tham số quan trọng](#11-các-siêu-tham-số-quan-trọng)
12. [Những hiểu lầm thường gặp](#12-những-hiểu-lầm-thường-gặp)
13. [Bài tập tự luyện](#13-bài-tập-tự-luyện)

---

## 1. Bài toán mà LoRA giải quyết

Hãy tưởng tượng một mô hình ngôn ngữ đã học được một **cuốn bách khoa toàn thư khổng lồ**. Bây giờ ta chỉ muốn dạy nó một kỹ năng nhỏ, chẳng hạn:

- trả lời theo phong cách chăm sóc khách hàng;
- phân loại phản hồi tích cực và tiêu cực;
- hiểu thuật ngữ của một công ty;
- viết câu trả lời theo một định dạng cố định.

Nếu dùng **full fine-tuning**, ta cho phép gần như toàn bộ tham số của mô hình thay đổi. Cách này giống như in lại cả cuốn bách khoa chỉ để bổ sung vài trang hướng dẫn. Nó cần nhiều bộ nhớ GPU vì trong lúc huấn luyện, máy còn phải giữ gradient và trạng thái của optimizer cho rất nhiều tham số.

LoRA chọn một cách thông minh hơn:

> Giữ nguyên kiến thức cũ của mô hình và chỉ học một “bản vá” nhỏ mô tả cách mô hình cần thay đổi.

LoRA là viết tắt của **Low-Rank Adaptation** — thích nghi bằng cập nhật hạng thấp. Đây là một kỹ thuật thuộc nhóm **PEFT** (*Parameter-Efficient Fine-Tuning*), nghĩa là fine-tune chỉ với một phần nhỏ tham số.

### Bức tranh tổng quát

| Cách huấn luyện | Trọng số gốc | Phần được học |
|---|---|---|
| Full fine-tuning | Thay đổi | Toàn bộ hoặc gần toàn bộ mô hình |
| LoRA | Đóng băng | Hai ma trận nhỏ `A` và `B` |

LoRA không làm mô hình gốc nhỏ đi. Nó chủ yếu làm **phần cần huấn luyện** nhỏ đi rất nhiều.

---

## 2. Ôn lại lớp tuyến tính `nn.Linear`

Transformer chứa rất nhiều lớp tuyến tính. Trong PyTorch, một lớp như vậy thường được tạo bằng:

```python
layer = nn.Linear(in_features=3, out_features=2)
```

Nó nhận một vector gồm 3 số và tạo ra một vector gồm 2 số.

### Công thức

Với một mẫu đầu vào, ta có:

$$
y = Wx + b
$$

Trong đó:

- $x$: đầu vào;
- $W$: ma trận trọng số;
- $b$: độ lệch (*bias*);
- $y$: đầu ra.

PyTorch thường lưu một batch theo dạng các vector hàng, vì vậy khi nhìn vào mã, ta sẽ thấy phép tính tương đương:

$$
Y = XW^T + b
$$

### Hình dạng của dữ liệu

Giả sử:

```text
X: [batch, in_features]  = [1, 3]
W: [out_features, in_features] = [2, 3]
b: [out_features] = [2]
Y: [batch, out_features] = [1, 2]
```

Ví dụ cụ thể:

$$
X = \begin{bmatrix}1 & 2 & 3\end{bmatrix}
$$

$$
W = \begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.4 & 0.5 & 0.6
\end{bmatrix}, \qquad
b = \begin{bmatrix}0.7 & 0.8\end{bmatrix}
$$

Đầu ra thứ nhất:

$$
1(0.1) + 2(0.2) + 3(0.3) + 0.7 = 2.1
$$

Đầu ra thứ hai:

$$
1(0.4) + 2(0.5) + 3(0.6) + 0.8 = 4.0
$$

Vậy:

$$
Y = \begin{bmatrix}2.1 & 4.0\end{bmatrix}
$$

> Lưu ý: kết quả $4.0$ là phép tính đúng với các con số ở trên. Khi học ma trận, nên tự tính lại thay vì tin ngay vào kết quả được in sẵn.

Mã kiểm tra:

```python
import torch
import torch.nn as nn

layer = nn.Linear(3, 2, bias=True)

with torch.no_grad():
    layer.weight.copy_(torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]))
    layer.bias.copy_(torch.tensor([0.7, 0.8]))

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print(y)        # tensor([[2.1000, 4.0000]])
print(y.shape)  # torch.Size([1, 2])
```

### Vì sao lớp tuyến tính trở nên đắt đỏ?

Một ma trận `4096 × 4096` có:

$$
4096 \times 4096 = 16{,}777{,}216
$$

trọng số. Đây mới chỉ là **một ma trận**. Một LLM có nhiều tầng, và mỗi tầng lại có nhiều lớp tuyến tính. Nếu cập nhật tất cả, chi phí bộ nhớ sẽ tăng rất nhanh.

---

## 3. Ý tưởng cốt lõi của LoRA

Gọi ma trận đã được pre-train là $W_0$. Full fine-tuning trực tiếp sửa $W_0$, còn LoRA giữ nguyên nó và học một phần thay đổi $\Delta W$:

$$
W_{\text{mới}} = W_0 + \Delta W
$$

Nếu $\Delta W$ lớn bằng $W_0$, ta vẫn chưa tiết kiệm được gì. Vì vậy LoRA biểu diễn nó bằng tích của hai ma trận nhỏ:

$$
\Delta W = \frac{\alpha}{r}BA
$$

Suy ra đầu ra của lớp LoRA:

$$
y = W_0x + \frac{\alpha}{r}B(Ax) + b
$$

Trong đó:

- $W_0$: trọng số gốc, bị đóng băng;
- $A$ và $B$: hai ma trận được huấn luyện;
- $r$: **rank** LoRA, kích thước của “nút thắt cổ chai”;
- $\alpha$: hệ số điều khiển độ mạnh của nhánh LoRA;
- $\alpha/r$: hệ số scale thường dùng.

### Một phép so sánh đời thường

Hãy hình dung $W_0$ là một giáo viên đã có kiến thức phổ thông. Muốn giáo viên đó dạy theo phong cách luyện thi, ta không xóa toàn bộ trí nhớ của họ. Ta chỉ đưa thêm một cuốn sổ mỏng chứa những điều chỉnh cần thiết.

- Mô hình gốc: kiến thức đã có.
- Hai ma trận `A`, `B`: cuốn sổ điều chỉnh.
- Fine-tuning LoRA: viết vào cuốn sổ, không sửa cả bộ não.

### Tại sao lại là hai ma trận?

Giả sử $W_0$ có kích thước `4096 × 4096`. Thay vì học hơn 16 triệu số, ta chọn $r=8$:

```text
A: [8, 4096]
B: [4096, 8]
```

Khi nhân `B @ A`, kết quả vẫn có kích thước:

```text
[4096, 8] @ [8, 4096] -> [4096, 4096]
```

Hai ma trận nhỏ có thể tạo ra một bản cập nhật đúng hình dạng của ma trận gốc, nhưng chỉ cần lưu ít số hơn rất nhiều.

---

## 4. Theo dõi hình dạng tensor

Đây là phần quan trọng nhất nếu bạn thường bị rối bởi kích thước tensor.

Giả sử đầu vào có:

```text
batch_size   = B
in_features  = I
out_features = O
rank         = R
```

Các tensor có hình dạng:

| Tensor | Shape | Vai trò |
|---|---:|---|
| `x` | `[B, I]` | Dữ liệu đầu vào |
| `W0` | `[O, I]` | Trọng số gốc |
| `A` | `[R, I]` | Nén từ `I` chiều xuống `R` chiều |
| `B` | `[O, R]` | Mở từ `R` chiều lên `O` chiều |
| `base_output` | `[B, O]` | Kết quả từ mô hình gốc |
| `lora_output` | `[B, O]` | Phần điều chỉnh của LoRA |
| `output` | `[B, O]` | Tổng hai nhánh |

Trong PyTorch, `F.linear(x, A)` thực hiện `x @ A.T`:

```text
x @ A.T
[B, I] @ [I, R]
      -> [B, R]
```

Sau đó:

```text
(x @ A.T) @ B.T
[B, R] @ [R, O]
            -> [B, O]
```

Cuối cùng, hai nhánh có cùng hình dạng nên cộng được với nhau:

```text
base_output + lora_output
   [B, O]   +    [B, O]
              -> [B, O]
```

Nhánh LoRA tạo thành một chiếc “đồng hồ cát”:

```text
I chiều -> R chiều -> O chiều
```

Thông thường `R` nhỏ hơn `I` và `O` rất nhiều.

### Nếu đầu vào có nhiều token thì sao?

Trong Transformer, `x` thường có shape:

```text
[batch, sequence_length, hidden_size]
```

Ví dụ `[2, 128, 4096]`. `F.linear` chỉ biến đổi chiều cuối, nên:

```text
[2, 128, 4096]
        qua A
-> [2, 128, 8]
        qua B
-> [2, 128, 4096]
```

Hai chiều `batch` và `sequence_length` được giữ nguyên.

---

## 5. Một ví dụ số nhỏ

Ta dùng một lớp có:

```text
in_features  = 3
out_features = 4
rank R       = 2
alpha        = 4
scale        = alpha / R = 2
```

Chọn:

$$
A = \begin{bmatrix}
1 & 0 & 2 \\
0 & 3 & 0
\end{bmatrix}
$$

$$
B = \begin{bmatrix}
1 & 0 \\
0 & 0 \\
0 & 2 \\
1 & 1
\end{bmatrix}
$$

Với đầu vào:

$$
x = \begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix}
$$

### Bước 1: `A` nén đầu vào

$$
Ax =
\begin{bmatrix}
1 & 0 & 2 \\
0 & 3 & 0
\end{bmatrix}
\begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix}
=
\begin{bmatrix}7 \\ 6\end{bmatrix}
$$

Vector 3 chiều đã trở thành vector 2 chiều.

### Bước 2: `B` mở rộng trở lại

$$
B(Ax) =
\begin{bmatrix}
1 & 0 \\
0 & 0 \\
0 & 2 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}7 \\ 6\end{bmatrix}
=
\begin{bmatrix}7 \\ 0 \\ 12 \\ 13\end{bmatrix}
$$

### Bước 3: Nhân hệ số scale

$$
\frac{\alpha}{r}B(Ax)
= 2
\begin{bmatrix}7 \\ 0 \\ 12 \\ 13\end{bmatrix}
=
\begin{bmatrix}14 \\ 0 \\ 24 \\ 26\end{bmatrix}
$$

Đây là phần điều chỉnh mà LoRA cộng vào kết quả của lớp gốc.

### Điều cần ghi nhớ

Trong lúc huấn luyện, ta tính lần lượt `x -> A -> B`. Ta **không cần tạo sẵn** ma trận lớn $\Delta W$ ở mỗi lần forward.

Sau khi huấn luyện, nếu muốn triển khai dưới dạng một lớp tuyến tính bình thường, ta có thể gộp:

$$
W_{\text{mới}} = W_0 + \frac{\alpha}{r}BA
$$

Việc gộp giúp nhánh LoRA không tạo thêm phép tính riêng khi suy luận. Tuy nhiên, nếu muốn đổi qua lại giữa nhiều adapter, ta có thể giữ adapter tách rời.

---

## 6. LoRA tiết kiệm đến mức nào?

Với ma trận $W$ có shape `[O, I]`:

- Full fine-tuning học $O \times I$ trọng số.
- LoRA học $R \times I + O \times R = R(I+O)$ trọng số.

Với `I = O = 4096` và `R = 8`:

| Phương pháp | Số tham số cần học |
|---|---:|
| Full fine-tuning | `4096 × 4096 = 16,777,216` |
| LoRA | `8 × 4096 + 4096 × 8 = 65,536` |

Tỷ lệ tham số LoRA so với full fine-tuning:

$$
\frac{65{,}536}{16{,}777{,}216} \times 100\% \approx 0.39\%
$$

Nói cách khác, riêng ở ma trận này, số tham số cần học giảm khoảng **99.61%**.

> Đây là mức giảm số tham số trainable, không có nghĩa toàn bộ bộ nhớ huấn luyện cũng giảm đúng 99.61%. Ta vẫn cần nạp trọng số mô hình gốc và giữ activation của quá trình forward/backward.

---

## 7. Tự xây dựng `LoRALinear` bằng PyTorch

### Mã nguồn

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank phải lớn hơn 0")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)

        # Đóng băng cả weight và bias của lớp gốc.
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        # A: [rank, in_features]
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features)
        )

        # B: [out_features, rank]
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_output = F.linear(hidden, self.lora_B)
        lora_output = lora_output * self.scaling

        return base_output + lora_output
```

### Giải thích từng phần

#### `self.base_layer = base_layer`

Ta giữ lại lớp tuyến tính đã được pre-train. LoRA không xóa hoặc tạo lại kiến thức của lớp này.

#### Đóng băng tham số gốc

```python
for parameter in self.base_layer.parameters():
    parameter.requires_grad = False
```

`requires_grad = False` báo cho PyTorch rằng không cần tính gradient và không cập nhật tham số đó. Vòng lặp đóng băng cả `weight` lẫn `bias` nếu bias tồn tại.

#### Tại sao dùng `nn.Parameter`?

```python
self.lora_A = nn.Parameter(...)
self.lora_B = nn.Parameter(...)
```

Một tensor thông thường chỉ là dữ liệu. `nn.Parameter` đăng ký tensor đó là tham số của mô hình, nhờ vậy:

- xuất hiện trong `model.parameters()`;
- được lưu trong `state_dict`;
- có thể nhận gradient;
- có thể được optimizer cập nhật.

#### Tại sao `A` ngẫu nhiên nhưng `B` bằng 0?

Ban đầu:

$$
B = 0 \Rightarrow BA = 0
$$

Do đó:

$$
y = W_0x + 0
$$

Ngay trước khi bắt đầu học, lớp LoRA hoạt động giống hệt lớp gốc. Đây là một điểm khởi đầu ổn định.

Không nên khởi tạo cả `A` và `B` bằng 0. Nếu làm vậy, ở bước backward đầu tiên gradient của cả hai nhánh có thể bị kẹt ở 0, khiến chúng không học được.

#### Hai lần gọi `F.linear`

```python
hidden = F.linear(x, self.lora_A)
lora_output = F.linear(hidden, self.lora_B)
```

Tương đương:

```python
hidden = x @ self.lora_A.T
lora_output = hidden @ self.lora_B.T
```

Ta đi qua không gian nhỏ `rank` thay vì tạo ma trận cập nhật lớn.

#### Dropout chỉ nằm ở nhánh LoRA

`lora_dropout` có thể giúp giảm overfitting khi dữ liệu fine-tune ít. Khi `dropout=0.0`, lớp này không làm thay đổi dữ liệu.

---

## 8. Gắn LoRA vào một mô hình

Ta có thể duyệt từng module con và thay `nn.Linear` bằng `LoRALinear`.

```python
def inject_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_names: tuple[str, ...] | None = None,
) -> None:
    """Thay các lớp Linear phù hợp bằng LoRALinear ngay trong mô hình."""

    for child_name, child_module in list(module.named_children()):
        full_name_matches = (
            target_names is None
            or any(target in child_name for target in target_names)
        )

        if isinstance(child_module, nn.Linear) and full_name_matches:
            setattr(
                module,
                child_name,
                LoRALinear(
                    child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
        else:
            inject_lora(
                child_module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_names=target_names,
            )
```

Hàm này duyệt theo quan hệ cha–con nên dùng được cả với các lớp nằm trực tiếp trong `nn.Sequential`.

### Gắn vào tất cả lớp tuyến tính

```python
inject_lora(model, rank=8, alpha=16)
```

### Chỉ gắn vào một số lớp có tên phù hợp

Trong mô hình Transformer thực tế, ta thường chọn đích cụ thể:

```python
inject_lora(
    model,
    rank=8,
    alpha=16,
    target_names=("q_proj", "v_proj"),
)
```

> Hàm tự viết ở đây dùng để học nguyên lý. Với mô hình thật, thư viện PEFT thường xử lý thêm nhiều trường hợp đặc biệt như chia sẻ trọng số, lượng tử hóa, lưu adapter và merge/unmerge.

---

## 9. Chương trình hoàn chỉnh có thể chạy

### Cài đặt

```bash
pip install torch
```

### Mã demo

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank phải lớn hơn 0")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base_layer(x)
        hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_output = F.linear(hidden, self.lora_B)
        return base_output + lora_output * self.scaling


def inject_lora(module, rank=8, alpha=16.0, dropout=0.0):
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear):
            setattr(
                module,
                child_name,
                LoRALinear(child_module, rank, alpha, dropout),
            )
        else:
            inject_lora(child_module, rank, alpha, dropout)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# 1. Tạo mô hình đồ chơi.
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# 2. Đóng băng toàn bộ trước khi gắn LoRA.
# Việc này đặc biệt hữu ích nếu mô hình còn các loại lớp khác.
for parameter in model.parameters():
    parameter.requires_grad = False

# 3. Gắn LoRA.
inject_lora(model, rank=8, alpha=16.0, dropout=0.05)

# 4. Chỉ lấy những tham số được phép học.
trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)
loss_function = nn.CrossEntropyLoss()

# 5. Tạo một batch giả.
x = torch.randn(4, 128)                 # [batch=4, features=128]
labels = torch.tensor([0, 1, 2, 3])     # 4 nhãn thuộc 10 lớp

# 6. Một bước huấn luyện.
optimizer.zero_grad()
logits = model(x)                       # [4, 10]
loss = loss_function(logits, labels)
loss.backward()
optimizer.step()

# 7. Kiểm tra kết quả.
total, trainable = count_parameters(model)

print("Shape đầu ra:", logits.shape)
print("Loss:", loss.item())
print("Tổng số tham số:", total)
print("Số tham số được học:", trainable)
print("Tỷ lệ trainable:", f"{100 * trainable / total:.2f}%")

print("\nCác tham số được optimizer cập nhật:")
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(f"- {name}: {list(parameter.shape)}")
```

Kết quả quan trọng không phải là giá trị `loss` ngẫu nhiên, mà là danh sách tham số được học. Bạn chỉ nên thấy các tensor có tên `lora_A` và `lora_B`.

### Tại sao demo đóng băng toàn bộ mô hình trước?

`LoRALinear` đã tự đóng băng lớp `Linear` mà nó bọc. Tuy nhiên, mô hình thực tế còn có embedding, LayerNorm hoặc các tham số khác. Đóng băng toàn bộ trước rồi để `A`, `B` mới tạo ra được train giúp quy tắc trở nên rõ ràng:

```text
Mô hình cũ: không học
Adapter LoRA mới: được học
```

---

## 10. LoRA nằm ở đâu trong Transformer?

Trong một khối Transformer, LoRA thường được gắn vào các phép chiếu tuyến tính.

### Attention

Attention tạo ra Query, Key và Value:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Các tên thường gặp trong mã nguồn:

- `q_proj`: tạo Query;
- `k_proj`: tạo Key;
- `v_proj`: tạo Value;
- `o_proj`: chiếu đầu ra attention.

Một cấu hình phổ biến là bắt đầu với `q_proj` và `v_proj`. Tuy vậy, đây không phải luật bắt buộc. Tùy mô hình, dữ liệu và ngân sách, người ta có thể gắn thêm `k_proj`, `o_proj` hoặc các lớp MLP.

### Feed-Forward Network

Khối MLP cũng chứa các lớp tuyến tính lớn. Ở những mô hình họ Llama, bạn có thể gặp:

- `gate_proj`;
- `up_proj`;
- `down_proj`.

Gắn LoRA vào nhiều vị trí hơn thường tăng khả năng thích nghi, nhưng cũng làm tăng số tham số trainable và chi phí huấn luyện.

---

## 11. Các siêu tham số quan trọng

| Siêu tham số | Ý nghĩa | Gợi ý để bắt đầu |
|---|---|---|
| `rank` hoặc `r` | Kích thước không gian trung gian | Thử `8` hoặc `16` |
| `alpha` | Độ mạnh của cập nhật LoRA | Thường thử `16` hoặc `32` |
| `dropout` | Giảm overfitting ở nhánh LoRA | Thử `0.0` đến `0.1` |
| `target_modules` | Những lớp sẽ nhận LoRA | Bắt đầu với `q_proj`, `v_proj` |
| `learning_rate` | Tốc độ cập nhật `A`, `B` | Phải thử nghiệm theo bài toán |

### Rank lớn hơn có luôn tốt hơn không?

Không. Rank lớn cho LoRA nhiều khả năng biểu diễn hơn nhưng:

- tốn thêm bộ nhớ;
- huấn luyện chậm hơn;
- có thể overfit nếu dữ liệu ít;
- không đảm bảo chất lượng tăng tương ứng.

Hãy xem `rank` như số dòng trong cuốn sổ điều chỉnh. Quá ít dòng thì không ghi đủ; quá nhiều dòng thì tốn kém và có thể ghi cả những chi tiết nhiễu.

### Vai trò của `alpha`

LoRA thường dùng hệ số:

$$
\text{scaling} = \frac{\alpha}{r}
$$

Nếu `alpha=16`, `r=8`, scale bằng `2`. Nếu `alpha=8`, `r=8`, scale bằng `1`. `alpha` không phải “độ chính xác”; nó chỉ ảnh hưởng độ lớn của phần cập nhật.

---

## 12. Những hiểu lầm thường gặp

### “LoRA nén toàn bộ mô hình”

Không hẳn. Trọng số gốc vẫn phải tồn tại để chạy mô hình. LoRA làm nhỏ phần **được huấn luyện** và file adapter, chứ không tự biến một mô hình lớn thành mô hình nhỏ.

### “Chỉ cần lưu `A` và `B`, không cần mô hình gốc”

Sai. Adapter chỉ là phần thay đổi. Khi chạy, bạn vẫn cần đúng base model tương thích, trừ khi đã merge và lưu thành một mô hình hoàn chỉnh.

### “LoRA luôn tốt ngang full fine-tuning”

Không có đảm bảo tuyệt đối. LoRA thường rất hiệu quả, nhưng kết quả còn phụ thuộc dữ liệu, loại nhiệm vụ, vị trí gắn adapter, rank và cách huấn luyện.

### “Đóng băng tham số nghĩa là lớp đó không chạy”

Sai. Lớp gốc vẫn tham gia forward để tạo đầu ra. “Đóng băng” chỉ có nghĩa là trọng số của nó không được optimizer thay đổi.

### “Merge làm mất base model”

Phép merge tạo $W_{\text{mới}} = W_0 + \Delta W$. Nếu cần giữ khả năng đổi adapter, hãy giữ một bản base model và các adapter riêng biệt.

### “LoRA và QLoRA là một”

Không. LoRA học adapter hạng thấp. QLoRA còn lượng tử hóa mô hình gốc, thường xuống 4-bit, để giảm thêm bộ nhớ. Adapter LoRA vẫn thường được huấn luyện ở độ chính xác cao hơn.

---

## 13. Bài tập tự luyện

### Bài 1 — Theo dõi shape

Cho:

```text
x: [16, 512]
W: [1024, 512]
rank: 4
```

Hãy xác định shape của `A`, `B`, `x @ A.T` và đầu ra cuối.

<details>
<summary>Xem đáp án</summary>

```text
A: [4, 512]
B: [1024, 4]
x @ A.T: [16, 4]
đầu ra LoRA: [16, 1024]
đầu ra cuối: [16, 1024]
```

</details>

### Bài 2 — Đếm tham số

Một lớp có `in_features=2048`, `out_features=4096`, LoRA rank `r=16`. Hãy tính số tham số của full fine-tuning và LoRA, bỏ qua bias.

<details>
<summary>Xem đáp án</summary>

Full fine-tuning:

$$
4096 \times 2048 = 8{,}388{,}608
$$

LoRA:

$$
16 \times 2048 + 4096 \times 16 = 98{,}304
$$

</details>

### Bài 3 — Quan sát khởi tạo

Trước khi huấn luyện, hãy so sánh đầu ra của một `nn.Linear` với `LoRALinear` bọc quanh chính lớp đó. Hai đầu ra phải giống nhau vì `lora_B` ban đầu bằng 0.

Gợi ý:

```python
torch.allclose(base_output, lora_output)
```

### Bài 4 — Kiểm tra gradient

Sau `loss.backward()`, hãy in:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.grad is None)
```

Giải thích vì sao trọng số gốc không có gradient, còn các tham số LoRA có thể có gradient.

### Bài 5 — Thử các rank khác nhau

Chạy demo với `rank = 2, 4, 8, 16`. Ghi lại:

- số tham số trainable;
- thời gian mỗi bước huấn luyện;
- loss sau cùng trên cùng một bộ dữ liệu.

Từ đó tự trả lời: rank nào tạo cân bằng tốt nhất cho bài toán của bạn?

---

## Tóm tắt trong 6 dòng

1. LLM có nhiều ma trận rất lớn; cập nhật tất cả chúng rất tốn tài nguyên.
2. LoRA đóng băng trọng số gốc $W_0$.
3. Phần thay đổi được biểu diễn bằng $\Delta W = (\alpha/r)BA$.
4. `A` nén dữ liệu xuống `r` chiều, `B` mở rộng trở lại.
5. Chỉ `A` và `B` nhận cập nhật từ optimizer.
6. Khi cần, adapter có thể được giữ riêng hoặc merge vào trọng số gốc để suy luận.

Nếu chỉ nhớ một câu, hãy nhớ câu này:

> **LoRA không bắt mô hình học lại mọi thứ; nó chỉ dạy mô hình cách điều chỉnh kiến thức đã có bằng một bản cập nhật nhỏ.**

---

## Gợi ý bước học tiếp theo

Sau khi hiểu và tự chạy được mã trong README này, bạn có thể học tiếp theo thứ tự:

1. dùng Hugging Face `transformers` để nạp một mô hình nhỏ;
2. dùng thư viện `peft` để gắn LoRA;
3. chuẩn bị dữ liệu dạng instruction–response;
4. fine-tune và lưu riêng adapter;
5. nạp lại, đánh giá và thử merge adapter;
6. tìm hiểu QLoRA khi GPU có bộ nhớ hạn chế.

