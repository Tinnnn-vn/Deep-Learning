# Từ Nhiễu Ngẫu Nhiên Đến Hình Ảnh (Giải Thích Diffusion)
### I. Hiểu về Diffusion Model bằng PyTorch cho người mới bắt đầu
Bạn đã bao giờ tự hỏi vì sao một mô hình AI có thể tạo ra một bức ảnh mới chỉ từ một câu mô tả, hoặc thậm chí từ một vùng nhiễu ngẫu nhiên?

Bạn đã từng nghe nói về Diffusion? Bạn đã thấy DALL-E, Midjourney hay FLUX tạo ra những hình ảnh tuyệt đẹp chỉ từ một câu lệnh văn bản. Nhìn bên ngoài, việc tạo ảnh bằng AI có vẻ giống “phép màu”.

Nhưng đó không phải là phép thuật. Cơ chế cốt lõi đằng sau các mô hình này là "Mô Hình Xác Suất Khuếch Tán Khử nhiễu (DDPM)", và ý tưởng chính của nó lại đơn giản đến bất ngờ:

    Nếu ta có thể dạy mô hình cách biến một bức ảnh bị nhiễu trở lại thành ảnh rõ ràng, thì ta cũng có thể bắt đầu từ nhiễu ngẫu nhiên và dần dần biến nó thành một hình ảnh mới.

Nói đơn giản, Diffusion Model học quá trình khử nhiễu.

### 1.1 Ý tưởng cốt lõi
Hãy tưởng tượng ta có một ly nước lọc trong suốt. Ban đầu, ly nước sạch và không màu. Sau đó, ta nhỏ một giọt màu xanh vào ly nước. 
Ngay khoảnh khắc đầu tiên, giọt màu xanh vẫn còn khá tập trung ở một điểm. Nhưng chỉ vài giây sau, màu xanh bắt đầu lan ra xung quanh, rồi dần dần hòa toàn bộ vào ly nước.

Diffusion là quá trình một thứ đang rõ ràng, có cấu trúc, dần dần bị lan ra và trở nên hỗn loạn hơn.

Muốn gom màu xanh lại thành một giọt ban đầu, ta thực hiện quá trình khử nhiễu. Ngoài đời thật, khi màu xanh đã lan ra khắp ly nước, gần như ta không thể tự nhiên gom màu xanh lại thành một giọt ban đầu.

Nhưng trong AI, ta sẽ huấn luyện một mô hình để học cách làm điều ngược lại.
Tức là mô hình sẽ học:

Từ một ảnh bị nhiễu, làm sao dự đoán phần nhiễu cần loại bỏ?

<img width="1122" height="1402" alt="ChatGPT Image Jun 27, 2026, 11_08_47 AM" src="https://github.com/user-attachments/assets/64ce5708-c9bc-4753-ae53-2db7b0403b8d" />

### 1.2 Ý tưởng này liên quan gì đến AI tạo ảnh?
Trong Diffusion Model, ta cũng làm một việc tương tự. Nhưng thay vì nhỏ màu xanh vào nước, ta thêm nhiễu ngẫu nhiên vào ảnh.

Hãy tưởng tượng ta có một bức ảnh rõ ràng, ví dụ ảnh một "con mèo".

Ban đầu ảnh rất rõ -> Sau đó ta thêm một chút nhiễu (nhiễu Gaussian) -> Ảnh vẫn là "con mèo", nhưng hơi mờ và có các chấm nhiễu -> Rồi thêm nhiều nhiễu hơn -> "con mèo" bắt đầu khó nhìn hơn. Cuối cùng, sau rất nhiều bước thêm nhiễu -> Ảnh gần như chỉ còn là một đống nhiễu ngẫu nhiên. Quá trình này được gọi là Lan Truyền Tiến (Forward Process). 

Nó giống như việc giọt màu xanh dần lan ra trong ly nước. Mô hình sẽ học:

    Từ một ảnh nhiễu, làm sao để dự đoán được phần nhiễu cần loại bỏ?
Nếu mô hình đoán được nhiễu, ta có thể trừ nhiễu đó ra để ảnh rõ hơn một chút. Lặp lại quá trình này nhiều lần, ta có thể biến một ảnh toàn nhiễu thành một ảnh mới. Quá trình này gọi là Lan Truyền Ngược (Reverse Process)

<img width="1723" height="417" alt="forward-process" src="https://github.com/user-attachments/assets/0971412d-a533-4313-9671-7994781631a2" />

Diffusion Model không tạo ảnh ngay lập tức. Nó tạo ảnh bằng nhiều bước nhỏ. Trong lúc huấn luyện, ta dạy mô hình nhìn một ảnh đã bị nhiễu và đoán xem nhiễu trong ảnh là gì.

Tóm lại Diffusion Model gồm hai quá trình:
1. Quá trình Khuếch Tán Thuận (forward diffusion): dần dần thêm nhiễu vào dữ liệu đầu vào
2. Quá trình Khử Nhiễu Ngược (reverse denoising): học cách tạo ra dữ liệu thông qua việc khử nhiễu

Các khái niệm & công thức toán học mà bạn sẽ cần nắm vững trong hướng dẫn này:

- Khuếch Tán Thuận (forward diffusion): $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$

- Khử Nhiễu Ngược (reverse denoising): $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}t}} \epsilon\theta(x_t, t) \right) + \sigma_t z$$

- Dự đoán nhiễu (Noise Prediction): $$\epsilon_\theta(x_t, t)$$

- Loss thường dùng là Mean Squared Error (MSE): $$L = \mathbb{E}{x_0, t, \epsilon} \left[ | \epsilon - \epsilon\theta(x_t, t) |^2 \right]$$

- Kiến trúc SimpleUNet, SinusoidalPositionEmbeddings và cách đưa thông tin thời gian vào mô hình (conditioning).

Nhìn các công thức toán học có lẽ bạn sẽ cảm thấy khô khan và bối rối.

ĐỪNG LO! Chúng ta sẽ cùng mổ xẻ từng phần một của công thức & mã PyTorch trong các chương sau nhé.

### II. Viết mã bằng PyTorch
### 2.1 import thư viện:
Đây là các thư viện Deep Learning trong Python.
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị đang dùng:", device)
```
| Thành phần | Nó làm gì? | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `torch` | Thư viện chính của PyTorch, dùng để tạo tensor và tính toán. | Đây là "bộ não toán học" của chương trình. |
| `nn` | Chứa các lớp để xây dựng neural network. | Sau này ta dùng để tạo mô hình dự đoán nhiễu. |
| `DataLoader` | Chia dataset thành từng batch nhỏ. | Giống như người phát từng xấp ảnh cho mô hình học. |
| `datasets` | Cung cấp các bộ dữ liệu phổ biến như MNIST, CIFAR-10. | Giúp ta tải dữ liệu nhanh mà không cần tự chuẩn bị thủ công. |
| `transforms` | Biến đổi ảnh trước khi đưa vào mô hình. | Dùng để đổi ảnh thành tensor và chuẩn hóa pixel. |
| `matplotlib.pyplot` | Dùng để vẽ và hiển thị ảnh. | Giúp ta nhìn thấy ảnh gốc và ảnh bị thêm nhiễu. |
| `device` | Dùng GPU nếu có. | Chạy nhanh hơn khi huấn luyện mô hình. |

