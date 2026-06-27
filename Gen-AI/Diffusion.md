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
Đây là các thư viện Deep Learning trong Python. Chúng ta sẽ sử dụng bộ dữ liệu MNIST để thực hành.
```python
# config.py
import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
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

### 2.2 Bảng thiết kế:
Mọi dự án phức tạp, từ xây dựng cho đến mạng thần kinh, đều bắt đầu bằng một bản thiết kế. Bản thiết kế này xác định các tham số và kích thước then chốt đóng vai trò định hướng cho toàn bộ quá trình xây dựng.
Trước khi xây dựng mô hình Diffusion, ta cần định nghĩa các tham số quan trọng như kích thước ảnh, số bước diffusion, batch size, learning rate, số epoch,.v.v.

Thay vì viết các biến này rải rác khắp nơi, ta gom chúng vào một class tên là DiffusionConfig để code gọn hơn và dễ sửa hơn.

```python
@dataclass
class DiffusionConfig:
    image_size: int = 32
    in_channels: int = 1
    base_channels: int = 32
    time_emb_dim: int = 256
    timesteps: int = 300
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 2e-4
    num_sample_images: int = 16
    max_batches_per_epoch: int = 300
    data_dir: str = "data"
    output_dir: str = "outputs"
```

Lớp `@dataclass` trong Python, một cách đơn giản và gọn gàng để nhóm các biến lại với nhau. Nó là một vùng chứa tất cả các tham số mà chúng ta có thể điều chỉnh để thay đổi kích thước, tốc độ và hành vi của mô hình khuếch tán.

Trước khi viết bất kỳ dòng mã nào tiếp, chúng ta cần giải thích các tham số cốt lõi này. Hãy cùng xem xét từng tham số một.
| Tham số | Nó điều khiển điều gì? | Trực giác dễ hiểu | Giá trị hiện tại |
| :--- | :--- | :--- | :--- |
| `image_size` | Kích thước ảnh đầu vào | Kích thước tấm canvas ảnh | `32` |
| `in_channels` | Số kênh màu của ảnh | MNIST là ảnh trắng đen nên có 1 kênh | `1` |
| `base_channels` | Số channel ban đầu trong U-Net | Mô hình càng rộng thì càng học mạnh hơn nhưng chạy chậm hơn | `32` |
| `time_emb_dim` | Kích thước vector time embedding | Kích thước "đồng hồ thời gian" đưa vào mô hình | `256` |
| `timesteps` | Tổng số bước thêm nhiễu | Số bước màu xanh lan dần vào ly nước | `300` |
| `beta_start` | Lượng nhiễu ở bước đầu | Lúc đầu chỉ thêm một chút nhiễu | `0.0001` |
| `beta_end` | Lượng nhiễu ở bước cuối | Về cuối nhiễu mạnh hơn | `0.02` |
| `batch_size` | Số ảnh xử lý mỗi lần | Mỗi lần mô hình nhìn 64 ảnh | `64` |
| `epochs` | Số vòng huấn luyện toàn bộ dataset | Mô hình học đi học lại dữ liệu nhiều lần | `5` |
| `learning_rate` | Tốc độ cập nhật trọng số | Bước học của mô hình | `0.0002` |
| `num_sample_images` | Số ảnh sinh ra sau mỗi epoch | Dùng để quan sát model học đến đâu | `16` |
| `max_batches_per_epoch` | Giới hạn số batch mỗi epoch | Giúp chạy nhanh hơn trên Colab khi học thử | `300` |
| `data_dir` | Thư mục lưu dataset | Nơi chứa MNIST | `"data"` |
| `output_dir` | Thư mục lưu ảnh kết quả | Nơi lưu ảnh sample và checkpoint | `"outputs"` |

Các tham số này là nền tảng cho toàn bộ mô hình của chúng ta.

DiffusionConfig giống như bản thiết kế trước khi xây nhà. Nếu bạn muốn thay đổi mô hình, bạn không cần đi tìm từng dòng code rải rác. Chỉ cần sửa trong DiffusionConfig.

### 2.3 Dataset - Load dữ liệu MNIST
Trong phần này, ta sẽ chuẩn bị dữ liệu ảnh MNIST để huấn luyện Diffusion Model.

MNIST là bộ dữ liệu gồm các ảnh chữ số viết tay từ 0 đến 9. Mỗi ảnh gốc có kích thước 28x28 pixel và là ảnh trắng đen.

Tuy nhiên, trong project này, ta sẽ resize ảnh từ 28x28 lên 32x32.

Lý do là U-Net của ta sẽ downsample ảnh nhiều lần. Với kích thước 32x32, quá trình giảm kích thước sẽ đẹp và đều hơn:

32 → 16 → 8 → 4 → 2

Nếu dùng ảnh 28x28, khi downsample nhiều lần, kích thước có thể trở thành số lẻ hoặc khó khớp khi nối skip connection.

```python
def get_mnist_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader
```
| Thành phần | Vai trò chính | Trực giác dễ hiểu | Kết quả
| :--- | :--- | :--- | :--- |
| `get_mnist_dataloader(config)` | Hàm load và chuẩn bị dữ liệu MNIST | Chuẩn bị "nguyên liệu" cho mô hình học |
| `transforms.Compose([...])` | Gom nhiều bước xử lý ảnh thành một pipeline | Ảnh đi qua nhiều trạm xử lý trước khi vào model | Một transform pipeline |
| `transforms.Resize(...)` | Đổi kích thước ảnh về `32x32` | Làm ảnh vừa với kiến trúc U-Net | Ảnh 32x32 |
| `transforms.ToTensor()` | Chuyển ảnh thành tensor PyTorch | Biến ảnh thành ma trận số | Pixel từ [0, 255] thành [0, 1] |
| `transforms.Lambda(...)` | Chuẩn hóa pixel về `[-1, 1]` | Đưa dữ liệu về khoảng dễ học hơn | Pixel nằm trong [-1, 1] |
| `datasets.MNIST(...)` | Tải dataset MNIST | Lấy dữ liệu chữ số viết tay | MNIST |
| `DataLoader(...)` | Chia dataset thành từng batch | Đưa dữ liệu cho mô hình theo từng nhóm ảnh |
### Sau khi tạo DataLoader, ta kiểm tra thử một batch:
```python
dataloader = get_mnist_dataloader(config)
images, labels = next(iter(dataloader))

print("Shape images:", images.shape)
print("Shape labels:", labels.shape)
print("Pixel min:", images.min().item())
print("Pixel max:", images.max().item())
```
Kết quả sẽ là:
```text
Shape images: torch.Size([64, 1, 32, 32])
Shape labels: torch.Size([64])
Pixel min: -1.0
Pixel max: 1.0
```
Shape labels là 64 nghĩa là mỗi ảnh có một nhãn tương ứng (ảnh 1 → label 7, ảnh 2 → label 3,.v.v.). Tuy nhiên, trong Diffusion Model cơ bản này, ta chưa dùng labels.

Lý do là mô hình của ta đang học sinh ảnh theo kiểu unconditional generation, tức là tạo ảnh không cần điều kiện nhãn.

Trong bài toán phân loại ảnh, label rất quan trọng. Nhưng trong Diffusion Model cơ bản, nhiệm vụ của mô hình không phải là phân loại ảnh mà là dự đoán nhiễu.
### 2.4 Khuếch Tán Thuận (forward diffusion): Giải thích toán học
Khuếch tán thuân là quá trình phá hủy một hình ảnh bằng cách thêm nhiễu, từng bước một. Dưới đây là công thức cho một bước đơn lẻ:

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon$$

| Thuật ngữ | Ý nghĩa kỹ thuật | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| $x_t$ | Ảnh bị nhiễu tại bước $t$ | Kết quả đầu ra nhiễu hơn một chút so với bước trước |
| $x_{t-1}$ | Ảnh thu được từ bước trước đó | Đối tượng dữ liệu mà ta đang tiến hành làm nhiễu |
| $\epsilon$ | Nhiễu Gaussian nguyên bản $\sim N(0, 1)$ | Thành phần nhiễu ngẫu nhiên thuần túy |
| $\beta_t$ | Phương sai của nhiễu tại bước $t$ | Lượng nhiễu sẽ được thêm vào (rất nhỏ, ví dụ: 0.0001 đến 0.02) |
| $\alpha_t = 1 - \beta_t$ | Tỷ lệ giữ lại tín hiệu ảnh gốc | Lượng thông tin của ảnh bước trước được giữ lại (gần bằng 1) |
| $\sqrt{\alpha_t}$ | Hệ số thu nhỏ (scale) của ảnh | Ta giảm nhẹ kích thước giá trị pixel của ảnh gốc... |
| $\sqrt{\beta_t}$ | Hệ số tỷ lệ (scale) của nhiễu | ...và cộng thêm vào một lượng nhiễu nhỏ tương ứng |

**Tại sao lại dùng căn bậc hai?** Chúng ta đang làm việc với *phương sai* (variance), chứ không phải *độ lệch chuẩn* (standard deviation). Khi bạn nhân một biến ngẫu nhiên với một hệ số $c$, phương sai của nó sẽ được nhân lên với $c^2$. Vì vậy, để thêm nhiễu với phương sai $\beta_t$, chúng ta phải nhân nó với hệ số $\sqrt{\beta_t}$. Việc sử dụng các căn bậc hai này nhằm đảm bảo tổng phương sai luôn được kiểm soát ở mức ổn định: $(\sqrt{\alpha_t})^2 + (\sqrt{\beta_t})^2 = \alpha_t + \beta_t = 1$

### Lịch trình phương sai - Variance Schedule ($\beta_t$)

Chìa khóa của quá trình khuếch tán tiến (forward process) chính là **lịch trình phương sai** (variance schedule), ký hiệu là $\beta_t$ (beta). Lịch trình này quy định chính xác lượng nhiễu mà chúng ta sẽ thêm vào tại mỗi bước thời gian $t$. Trong bài báo gốc về DDPM, đây là một lịch trình tuyến tính (linear schedule) đơn giản:

* Tại $t = 1$, chúng ta thêm một lượng nhiễu cực kỳ nhỏ: $\beta_1 = 0.0001$
* Tại $t = 1000$, chúng ta thêm nhiều nhiễu hơn: $\beta_{1000} = 0.02$
* Các giá trị của $\beta_2, \beta_3, \dots$ được phân bổ cách đều nhau giữa hai điểm mút này.

Điều này có nghĩa là chúng ta bắt đầu bằng việc chỉ thêm một lượng nhiễu nhẹ như "tiếng thì thầm", và tăng dần lượng nhiễu ở mỗi bước tiếp theo.

Từ $\beta_t$, chúng ta suy ra được $\alpha_t = 1 - \beta_t$. Nếu $\beta_t$ là tỷ lệ nhiễu, thì $\alpha_t$ chính là **tỷ lệ tín hiệu** (signal rate)—cho biết lượng thông tin của bức ảnh trước đó được giữ lại bao nhiêu. Vì $\beta_t$ luôn rất nhỏ, nên $\alpha_t$ sẽ luôn gần bằng 1 (ví dụ: 0.9999).

### Vấn đề: Quá trình này rất chậm

Để thu được một bức ảnh bị nhiễu $x_t$ từ ảnh gốc $x_0$, theo lý thuyết chúng ta sẽ phải áp dụng công thức tuần tự $t$ lần:

$$x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_t$$

Với $t = 500$, điều đó đồng nghĩa với việc phải thực hiện 500 thao tác tuần tự liên tiếp. Điều này sẽ khiến cho việc huấn luyện mô hình trở nên vô cùng chậm chạp và tốn thời gian.

### Lối đi tắt (The Shortcut)

Hãy cùng xây dựng một công thức giúp chúng ta "nhảy" trực tiếp từ ảnh gốc $x_0$ sang ảnh bị nhiễu $x_t$. Bắt đầu với công thức biến đổi 1 bước và khai triển nó:

$$x_1 = \sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_1$$

$$x_2 = \sqrt{\alpha_2}x_1 + \sqrt{\beta_2}\epsilon_2$$

Thay thế giá trị của $x_1$ vào phương trình tính $x_2$:

$$x_2 = \sqrt{\alpha_2}\left( \sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_1 \right) + \sqrt{\beta_2}\epsilon_2$$

$$= \sqrt{\alpha_1\alpha_2}x_0 + \sqrt{\alpha_2\beta_1}\epsilon_1 + \sqrt{\beta_2}\epsilon_2$$

Đây là điểm mấu chốt quan trọng: khi bạn cộng hai biến ngẫu nhiên Gaussian độc lập với nhau, kết quả thu được cũng là một biến Gaussian với phương sai bằng tổng phương sai của hai biến thành phần. Do đó, hai số hạng chứa nhiễu được gộp lại như sau:

$$\sqrt{\alpha_2\beta_1}\epsilon_1 + \sqrt{\beta_2}\epsilon_2 \sim \mathcal{N}(0, \alpha_2\beta_1 + \beta_2)$$

Vì $\beta_1 = 1 - \alpha_1$, chúng ta có thể rút gọn biểu thức phương sai: $\alpha_2\beta_1 + \beta_2 = \alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1\alpha_2$

Vì vậy, ta có thể viết lại phương trình thành: 

$$x_2 = \sqrt{\alpha_1\alpha_2}x_0 + \sqrt{1 - \alpha_1\alpha_2}\epsilon$$

---

Quy luật đã quá rõ ràng. Tiếp tục khai triển này cho đến bước thời gian $t$, ta thu được:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

Trong đó, $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$ chính là tích lũy (cumulative product).

Đây là phương trình quan trọng bậc nhất đối với quá trình khuếch tán tiến (forward process). Hãy cùng phân tích chi tiết các thành phần:

* $x_0$ là bức ảnh sạch, ảnh gốc ban đầu của chúng ta.
* $\epsilon$ là một mẫu nhiễu Gaussian nguyên bản duy nhất.
* $\bar{\alpha}_t$ (alpha-bar) là tích lũy: $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$

Số hạng $\bar{\alpha}_t$ cho biết lượng tín hiệu của ảnh gốc còn sót lại bao nhiêu tại bước thời gian $t$. Khi $t$ tăng dần, $\bar{\alpha}_t$ giảm dần từ $1.0$ về $0.0$—đồng nghĩa với việc tín hiệu ảnh gốc sẽ nhạt nhòa dần cho đến khi chỉ còn lại nhiễu hoàn toàn.

Công thức này thực chất chỉ là một phép nhân tổng trọng số: chúng ta lấy một lượng $\sqrt{\bar{\alpha}_t}$ của ảnh gốc và cộng thêm một lượng $\sqrt{1 - \bar{\alpha}_t}$ của nhiễu thuần túy. Khi $t$ nhỏ, chúng ta giữ lại hầu hết thông tin của bức ảnh. Khi $t$ lớn, chúng ta hầu như không giữ lại gì ngoài nhiễu.

"Lối đi tắt" này là yếu tố cốt lõi giúp quá trình huấn luyện đạt hiệu quả cao. Chúng ta có thể tạo ngay lập tức một mẫu huấn luyện $(x_t, \epsilon)$ cho bất kỳ bước thời gian $t$ ngẫu nhiên nào mà không cần phải thực hiện qua vòng lặp tuần tự. Trong phần tiếp theo, chúng ta sẽ chuyển công thức này trực tiếp thành mã nguồn PyTorch.

### Triển khai mã nguồn
Chúng ta cần triển khai công thức này trong PyTorch:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

Nhìn vào công thức này, chúng ta cần ba thành phần:

1. $\bar{\alpha}_t$ — tỷ lệ tín hiệu tích lũy, được tính toán trước cho tất cả các bước thời gian (timesteps).
2. $x_0$ — bức ảnh sạch (đầu vào).
3. $\epsilon$ — nhiễu Gaussian nguyên bản (chúng ta sẽ lấy mẫu ngẫu nhiên trực tiếp trong quá trình chạy).

Vì $\bar{\alpha}_t$ chỉ phụ thuộc vào bước thời gian (không phụ thuộc vào bức ảnh), chúng ta có thể tính toán trước nó một lần và tái sử dụng nhiều lần. Mã triển khai của chúng ta sẽ gồm hai phần:

1. Tính toán trước các lịch trình nhiễu ($\beta_t, \alpha_t, \bar{\alpha}_t$).
2. Hàm "lối đi tắt" (shortcut) để tạo ra một bức ảnh bị nhiễu $x_t$ từ bất kỳ ảnh gốc $x_0$ và bước thời gian $t$ nào được cho sẵn.

### Phần 1: Tính toán trước các Lịch trình trong `Diffusion.__init__`

Hãy nhớ lại rằng $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$, trong đó $\alpha_t = 1 - \beta_t$. Công việc chúng ta cần làm là:

1. Tạo lịch trình cho $\beta_t$ (các giá trị phân bổ cách đều nhau theo đường tuyến tính).
2. Tính $\alpha_t = 1 - \beta_t$.
3. Tính $\bar{\alpha}_t$ dưới dạng tích lũy của $\alpha$.

Vì đây là các hằng số cố định, chúng ta sẽ tính toán chúng một lần duy nhất khi khởi tạo mô hình (initialization):

```python
class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = SimpleUNet(config).to(config.device)

        # Tính toán trước các thành phần của lịch trình nhiễu
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
```
