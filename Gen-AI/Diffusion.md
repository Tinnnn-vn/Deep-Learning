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

Với bản thiết kế và dữ liệu đã được xác định, giờ đây chúng ta đã sẵn sàng xây dựng thành phần chính đầu tiên của mô hình: quá trình toán học để thêm nhiễu & loại bỏ nhiễu khỏi ảnh.

### 2.4 Khuếch Tán Thuận (forward diffusion): Giải thích toán học
Khuếch tán thuận là quá trình phá hủy một hình ảnh bằng cách thêm nhiễu, từng bước một. Dưới đây là công thức cho một bước đơn lẻ:

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
2. Hàm "noise_image" để tạo ra một bức ảnh bị nhiễu $x_t$ từ bất kỳ ảnh gốc $x_0$ và bước thời gian $t$ nào được cho sẵn.

### Phần 1: Tính toán trước các lịch trình nhiễu trong `Diffusion.__init__`

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

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
```
Vì sao dùng register_buffer?

beta, alpha, alpha_hat là các tensor quan trọng của Diffusion, nhưng chúng không phải tham số cần học. Model cần lưu chúng lại và đưa chúng lên GPU cùng model, nhưng optimizer không được cập nhật chúng.

Nói đơn giản:

register_buffer dùng để lưu các tensor cố định đi theo model.

### Phần 2: Triển khai hàm noise_image để thêm nhiễu
Hàm này hiện thực công thức: $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

Hàm này nhận một loạt ảnh sạch x và một loạt các bước thời gian tương ứng t, và trả về các ảnh bị nhiễu $$x_t$$

Hãy cùng xem đoạn mã sau.
```python
    def noise_images(self, x, t):
        # Thêm nhiễu vào hình ảnh ở bước thời gian t
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε
```
| Ký hiệu | Trong code | Ý nghĩa |
| :--- | :--- | :--- |
| $x_0$ | `x` | Ảnh gốc sạch |
| $t$ | `t` | Timestep |
| $\epsilon$ | `noise` | Nhiễu Gaussian |
| $\sqrt{\bar{\alpha}_t}$ | `sqrt_alpha_hat` | Lượng ảnh gốc còn giữ lại |
| $\sqrt{1 - \bar{\alpha}_t}$ | `sqrt_one_minus_alpha_hat` | Lượng nhiễu được thêm vào |
| $x_t$ | `x_t` | Ảnh sau khi bị thêm nhiễu |

| Dòng code | Nó làm gì? | Trực giác dễ hiểu | Shape ví dụ |
| :--- | :--- | :--- | :--- |
| `self.alpha_hat[t]` | Lấy $\bar{\alpha}_t$ theo timestep | Mỗi ảnh lấy mức nhiễu riêng | `[64]` |
| `torch.sqrt(...)` | Lấy căn bậc hai | Đúng theo công thức diffusion | `[64]` |
| `[:, None, None, None]` | Reshape để nhân với ảnh | Biến hệ số thành dạng broadcast được | `[64, 1, 1, 1]` |
| `torch.randn_like(x)` | Tạo noise cùng shape ảnh | Tạo nhiễu Gaussian | `[64, 1, 32, 32]` |
| `sqrt_alpha_hat * x` | Giữ lại một phần ảnh gốc | Phần hình ảnh còn rõ | `[64, 1, 32, 32]` |
| `sqrt_one_minus_alpha_hat * noise` | Thêm nhiễu vào ảnh | Phần màu xanh lan vào ly nước | `[64, 1, 32, 32]` |
| `return x_t, noise` | Trả về ảnh nhiễu và noise thật | Noise thật dùng làm đáp án training | — |

Nói đơn giản: noise_images() là hàm chủ động phá hỏng ảnh gốc bằng cách thêm nhiễu
Chúng ta đã hoàn tất quá trình khuếch tán thuận. Giờ đã có một phương pháp xác định và hiệu quả để lấy bất kỳ hình ảnh nào và tạo ra phiên bản nhiễu cho bất kỳ bước thời gian t nào. Với các mẫu huấn luyện này, chúng ta đã sẵn sàng định nghĩa quá trình khử nhiễu ngược và dạy mạng U-Net cách dự đoán nhiễu.

### 2.5 Quá trình khử nhiễu ngược và huấn luyện (Reverse Process & Training)
Chúng ta đã biết học cách phá hủy hình ảnh bằng cách thêm nhiễu trước đó. Giờ đây chúng ta sẽ học cách sáng tạo lại ảnh đó.

Quá trình đảo ngược là bắt đầu với nhiễu thuần túy ($$x_T$$) và khử nhiễu dần dần, từng bước một, cho đến khi ta có được một hình ảnh sạch ($$x_0$$).

**Bài toán: Đảo ngược điều không thể đảo ngược (Reversing the Irreversible)**

Quá trình khuếch tán tiến (forward process) thêm nhiễu tại mỗi bước bằng cách sử dụng một phân phối xác suất có điều kiện $q(x_t \mid x_{t-1})$. Để đảo ngược quá trình này, chúng ta cần phải tính toán được xác suất của bức ảnh ở bước trước đó khi biết bức ảnh hiện tại: $p(x_{t-1} \mid x_t)$.

**Nhưng chẳng phải nhiễu ở mỗi bước là độc lập sao? Tại sao việc đảo ngược lại khó khăn?**

Đúng vậy, mỗi thành phần nhiễu $\epsilon_t$ đều độc lập. Nhưng khi đi theo chiều ngược lại, bạn không thể biết *chính xác* thành phần nhiễu nào đã được thêm vào—bạn không thể chỉ đơn giản là trừ nó đi được. Phân phối điều kiện nghịch đảo $p(x_{t-1} \mid x_t)$ đòi hỏi phải tính tích phân trên toàn bộ các bức ảnh gốc có thể có, và điều này là bất khả thi về mặt toán học (intractable).

Thật không may, việc tính toán trực tiếp phân phối này là một bài toán bất khả thi. Nó đòi hỏi phải sử dụng toàn bộ tập dữ liệu cho mỗi một bước đơn lẻ, khiến cho chi phí tính toán trở nên vô tận.

**Giải pháp: Huấn luyện một mạng Neural Network để xấp xỉ nó.**

Nếu không thể tính toán trực tiếp bước nghịch đảo bằng công thức toán học, chúng ta có thể huấn luyện một mạng neural đủ mạnh để *học* một hàm xấp xỉ cho nó. Mục tiêu của chúng ta là tạo ra một mô hình nhận vào một bức ảnh bị nhiễu $x_t$ và chỉ cho chúng ta biết bức ảnh bớt nhiễu $x_{t-1}$ sẽ trông như thế nào.

### Tái tham số hóa để dự đoán nhiễu

Các tác giả của bài báo gốc về DDPM đã có một phát hiện mang tính đột phá. Thay vì huấn luyện mạng neural để dự đoán trực tiếp các pixel của bức ảnh bớt nhiễu hơn một chút $x_{t-1}$, việc tái tham số hóa (re-parameterize) bài toán mang lại hiệu quả và sự ổn định cao hơn rất nhiều.

Nhiệm vụ của mạng được đơn giản hóa một cách tối đa: **Thay vì dự đoán bức ảnh, hãy dự đoán nhiễu.**

Hãy nghĩ về công thức của quá trình khuếch tán tiến: 

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$

Phương trình này chứa cả ba thành phần cốt lõi: bức ảnh bị nhiễu $x_t$, ảnh gốc sạch $x_0$, và thành phần nhiễu $\epsilon$. Nếu chúng ta đã biết $x_t$ và bước thời gian $t$ (từ đó suy ra được $\bar{\alpha}_t$), và bằng cách nào đó chúng ta đoán được lượng nhiễu $\epsilon$ đã được thêm vào, chúng ta hoàn toàn có thể biến đổi lại công thức để đưa ra ước lượng cho bức ảnh gốc $x_0$.

Điều này thay đổi hoàn toàn cách tiếp cận bài toán. Mạng neural của chúng ta, ký hiệu là $\epsilon_\theta$ (epsilon-theta), sẽ chỉ đảm nhận một nhiệm vụ duy nhất:

* **Đầu vào (Input):** Một bức ảnh bị nhiễu $x_t$ và bước thời gian $t$ tương ứng.
* **Đầu ra (Output):** Dự đoán về thành phần nhiễu $\epsilon$ đã được dùng để tạo ra $x_t$.

### Hàm mục tiêu: Một phép so sánh đơn giản

Việc tái định hình bài toán này giúp hàm mất mát (loss function) — thước đo cho biết mô hình đang "sai" như thế nào — trở nên vô cùng đơn giản:

1. Trong quá trình huấn luyện, chúng ta chọn một bức ảnh sạch thực tế $x_0$ và một bước thời gian ngẫu nhiên $t$.
2. Chúng ta sử dụng hàm `noise_images` từ Phần 2 trong Chương 2.4. Hàm này sẽ trả về hai thứ: bức ảnh bị nhiễu $x_t$ và thành phần "nhiễu thực tế" ($\epsilon$) đã được dùng để tạo ra nó.
3. Chúng ta đưa $x_t$ và $t$ vào mạng neural để thu được "nhiễu dự đoán" ($\epsilon_\theta$).
4. Hàm mất mát đơn giản chỉ là sự khác biệt giữa nhiễu thực tế và nhiễu dự đoán.

Đối với dữ liệu hình ảnh, cách phổ biến nhất để đo lường sự khác biệt này là sử dụng **Sai số bình phương trung bình (Mean Squared Error - MSE)**.

Hàm mục tiêu huấn luyện của chúng ta là:

$$L = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]$$

Hãy cùng diễn giải công thức này sang ngôn ngữ thông thường:

* $\mathbb{E}[\dots]$ : "Tính trung bình trên nhiều ví dụ..."
* $x_0, t, \epsilon$ : "Nơi chúng ta lấy một bức ảnh thực tế $x_0$, một bước thời gian ngẫu nhiên $t$, và một nhiễu ngẫu nhiên $\epsilon$..."
* $\| \dots \|^2$ : "Tính Sai số bình phương trung bình giữa..."
* $\epsilon$ : "Thành phần nhiễu thực tế..."
* $\epsilon_\theta(x_t, t)$ : "Thành phần nhiễu được dự đoán bởi mô hình của chúng ta khi nó nhìn vào bức ảnh nhiễu $x_t$ tại bước thời gian $t$."

Chỉ vậy thôi. Toàn bộ quá trình huấn luyện được tóm gọn lại là: Đưa cho mô hình một bức ảnh bị hỏng và hỏi nó, "Tôi đã thêm vào loại nhiễu nào?". Lượng nhiễu dự đoán càng gần với nhiễu thực tế thì giá trị loss càng thấp. Trong phần tiếp theo, chúng ta sẽ xem cách triển khai mã nguồn của Khử nhiễu ngược và huấn luyện.

### Triển khai mã nguồn

```python
    def sample_timesteps(self, n):
        # Chọn bước thời gian ngẫu nhiên
        return torch.randint(low=1, high=self.config.timesteps, size=(n,), device=self.config.device)

    def forward(self, x):
        # Huấn luyện: Tính toán hàm mất mát (MSE giữa nhiễu dự đoán và nhiễu thực tế)
        # 1. Lấy mẫu các mốc thời gian (timesteps)
        t = self.sample_timesteps(x.shape[0])
        # 2. Tạo ảnh nhiễu và lấy nhiễu thực tế
        x_t, noise = self.noise_images(x, t)
        # 3. Dự đoán nhiễu (sử dụng U-Net)
        predicted_noise = self.model(x_t, t)
        # 4. Tính Loss
        return F.mse_loss(noise, predicted_noise)
```
Trước tiên ta hãy cùng xem xét hàm `sample_timesteps` này trước nhé.

Mục đích của nó là tạo ra một lô (batch) các mốc thời gian (timesteps) ngẫu nhiên. Trong đó, n là kích thước lô (ví dụ: 8).

Hàm `torch.randint` tạo ra một tensor một chiều gồm n số nguyên ngẫu nhiên. Giá trị thấp nhất (low) là 1 và giá trị cao nhất (high) là `self.config.timesteps` (tức là 1000).

Vì sao timestep phải ngẫu nhiên?

Trong quá trình huấn luyện, chúng ta muốn mô hình trở thành một bộ dự đoán nhiễu mạnh mẽ, có khả năng xử lý mọi mức độ nhiễu. Bằng cách lấy mẫu ngẫu nhiên giá trị t cho mỗi hình ảnh trong từng lô, chúng ta đảm bảo mô hình được tiếp xúc với các ví dụ thuộc toàn bộ dải mức độ nhiễu (từ t=1 đến t=999) và không bị quá khớp (overfit) với bất kỳ mức độ cụ thể nào.

Tại sao không sử dụng tất cả các mốc thời gian cho mỗi hình ảnh? Bạn có thể làm vậy, nhưng sẽ rất lãng phí. Việc huấn luyện trên cả 1000 mốc thời gian cho một hình ảnh đồng nghĩa với việc phải thực hiện 1000 lượt truyền xuôi (forward pass) mới có được một lần cập nhật gradient. Phương pháp lấy mẫu ngẫu nhiên mang lại độ bao phủ tương đương theo thời gian nhưng lại tiết kiệm tài nguyên tính toán gấp 1000 lần cho mỗi bước. Phương pháp hạ gradient ngẫu nhiên (stochastic gradient descent) hoạt động hiệu quả trong trường hợp này.

| Dòng code | Nó làm gì? | Trực giác |
| :--- | :--- | :--- |
| `def sample_timesteps(self, n):` | Tạo hàm lấy timestep ngẫu nhiên | Mỗi ảnh sẽ bị nhiễu ở mức khác nhau |
| `torch.randint(...)` | Sinh số nguyên ngẫu nhiên | Chọn timestep |
| `low=1` | Bắt đầu từ timestep 1 | Tránh timestep 0 quá sạch |
| `high=self.config.timesteps` | Giới hạn trên | Nếu `timesteps=300`, lấy từ 1 đến 299 |
| `size=(n,)` | Tạo `n` timestep | Mỗi ảnh trong batch có một timestep |
| `device=self.alpha_hat.device` | Tạo tensor trên đúng device | Tránh lỗi CPU/GPU |
| `.long()` | Chuyển sang kiểu integer long | Dùng để index tensor |

Hàm `forward` — Training loss

Phương thức `forward` là trọng tâm của quá trình huấn luyện. Nó nhận vào một lô (batch) các ảnh sạch `x` (từ tập dữ liệu) và tính toán một giá trị mất mát (loss) duy nhất để cập nhật toàn bộ trọng số của mô hình. Phương thức này tuân thủ hoàn hảo logic đã trình bày ở chương 2.5.

| Dòng code | Nó làm gì? | Trực giác | Shape ví dụ |
| :--- | :--- | :--- | :--- |
| `def forward(self, x):` | Nhận batch ảnh gốc | Ảnh sạch từ MNIST | `[64, 1, 32, 32]` |
| `t = self.sample_timesteps(x.shape[0])` | Chọn timestep cho từng ảnh | Mỗi ảnh bị phá ở mức khác nhau | `[64]` |
| `x_t, noise = self.noise_images(x, t)` | Tạo ảnh nhiễu và noise thật | Tạo bài toán cho model học | `[64, 1, 32, 32]` |
| `predicted_noise = self.model(x_t, t)` | U-Net dự đoán noise | Model đoán nhiễu trong ảnh | `[64, 1, 32, 32]` |
| `F.mse_loss(noise, predicted_noise)` | So sánh noise thật và noise dự đoán | Model đoán càng sai thì loss càng lớn | scalar |
| `return loss` | Trả loss cho training loop | Optimizer sẽ dùng loss để cập nhật model | scalar |

Nói đơn giản: Ta đưa cho model một ảnh bị nhiễu và hỏi: “Noise nào đã được thêm vào ảnh này?

Chúng ta đã xác định xong toàn bộ quy trình huấn luyện. Bước tiếp theo là mở "hộp đen" của `self.model` và tìm hiểu cách thức thực sự mà SimpleUNet đưa ra dự đoán.

### 2.6 Mạng nơ ron dự đoán nhiễu: `SimpleUNet`
Trong Diffusion Model, ta không huấn luyện mô hình để vẽ ảnh trực tiếp. Thay vào đó, ta huấn luyện mô hình để dự đoán nhiễu.

Cụ thể, mô hình U-Net của chúng ta nhận đầu vào: $$[x_t, t]$$

Mô hình U-Net cần dự đoán: $$\epsilon_\theta(x_t, t)$$

Chúng ta đã xác định rằng mình cần một mô hình có khả năng nhìn vào một bức ảnh bị nhiễu $x_t$ và dự đoán phần nhiễu $\epsilon_\theta$ đã được thêm vào trước đó. Kiến trúc được lựa chọn cho nhiệm vụ này là U-Net. Chương này sẽ giải thích tại sao U-Net lại là công cụ hoàn hảo cho công việc này, đồng thời đi sâu vào cấu trúc tổng quan và luồng dữ liệu của nó.

Hãy xem kiến trúc mã sau.

```python
class SimpleUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        base = config.base_channels

        down_channels = (base,
            base * 2,
            base * 4,
            base * 8,
            base * 16,
        )

        up_channels = (
            base * 16,
            base * 8,
            base * 4,
            base * 2,
            base,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.time_emb_dim),
            nn.Linear(config.time_emb_dim, config.time_emb_dim),
            nn.GELU(),
        )

        self.conv0 = nn.Conv2d(
            config.in_channels,
            down_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.downs = nn.ModuleList([
            Block(
                down_channels[i],
                down_channels[i + 1],
                config.time_emb_dim,
                up=False,
            )
            for i in range(len(down_channels) - 1)
        ])

        self.ups = nn.ModuleList([
            Block(
                up_channels[i],
                up_channels[i + 1],
                config.time_emb_dim,
                up=True,
            )
            for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(
            up_channels[-1],
            config.in_channels,
            kernel_size=1,
        )

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Thêm kết nối tắt (skip connection)
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)
```

| Thành phần | Vai trò chính | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `SimpleUNet` | Mô hình dự đoán nhiễu | Bộ não nhìn ảnh nhiễu và đoán noise |
| `config` | Chứa tham số mô hình | Lấy kích thước ảnh, số channel, time embedding |
| `base_channels` | Số channel nền | Độ rộng ban đầu của mô hình |
| `down_channels` | Số channel ở đường xuống | Càng xuống sâu, channel càng tăng |
| `up_channels` | Số channel ở đường lên | Càng lên gần ảnh gốc, channel càng giảm |
| `time_mlp` | Xử lý timestep thành time embedding | Tạo tín hiệu thời gian cho toàn bộ U-Net |
| `conv0` | Convolution đầu vào | Đổi ảnh gốc thành feature map ban đầu |
| `downs` | Danh sách các Block downsample | Đường encoder |
| `ups` | Danh sách các Block upsample | Đường decoder |
| `output` | Convolution cuối | Đưa feature map về số channel ảnh ban đầu |

**Giải thích phần base, down_channels, up_channels**
```python
        base = config.base_channels

        down_channels = (base,
            base * 2,
            base * 4,
            base * 8,
            base * 16,
        )

        up_channels = (
            base * 16,
            base * 8,
            base * 4,
            base * 2,
            base,
        )
```

| Thành phần | Nó làm gì? | Với `base = 32` |
| :--- | :--- | :--- |
| `base` | Số channel nền | `32` |
| `down_channels` | Channel ở encoder | `(32, 64, 128, 256, 512)` |
| `up_channels` | Channel ở decoder | `(512, 256, 128, 64, 32)` |

Trong encoder, kích thước ảnh giảm dần nhưng số channel tăng dần:

32x32, 32 channels

→ 16x16, 64 channels 

→ 8x8, 128 channels 

→ 4x4, 256 channels 

→ 2x2, 512 channels

Trong decoder, ta làm ngược lại:

2x2, 512 channels

→ 4x4, 256 channels

→ 8x8, 128 channels

→ 16x16, 64 channels

→ 32x32, 32 channels

**Vì sao ảnh nhỏ dần nhưng channel tăng dần?**

Khi ảnh nhỏ dần, mô hình không còn tập trung vào từng pixel nhỏ nữa. Nó học các đặc trưng tổng quát hơn.
- Tầng đầu: Mô hình học các cạnh, nét cong, chấm nhiễu nhỏ
- Tầng giữa: Mô hình học một phần nét của chữ số
- Tầng cuối: Mô hình học cấu trúc tổng quát của chữ số

Ảnh càng nhỏ thì mô hình càng nhìn tổng quát hơn. Channel càng nhiều thì mô hình có nhiều “ngăn nhớ” hơn để lưu đặc trưng.

**Giải thích phần `time_mlp` trong U-Net**
```python
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.time_emb_dim),
            nn.Linear(config.time_emb_dim, config.time_emb_dim),
            nn.GELU(),
        )
```

| Thành phần | Vai trò | Trực giác |
| :--- | :--- | :--- |
| `SinusoidalPositionEmbeddings(...)` | Biến timestep thành vector sin/cos | Tạo mã thời gian ban đầu |
| `nn.Linear(...)` | Học cách biến đổi vector thời gian | Cho model tự học cách dùng timestep |
| `nn.GELU()` | Hàm kích hoạt | Giúp tín hiệu thời gian linh hoạt hơn |
| `self.time_mlp` | Pipeline xử lý timestep | Biến `timestep` thành embedding dùng trong các Block |

Nếu timestep ban đầu có shape: `[64]`, thì sau time_mlp ta có: `[64, 256]`. Vector này sẽ được đưa vào tất cả các Block

**Giải thích phần Convolution đầu vào `conv0`**
```python
        self.conv0 = nn.Conv2d(
            config.in_channels,
            down_channels[0],
            kernel_size=3,
            padding=1,
        )
```

| Thành phần | Nó làm gì? | Với MNIST |
| :--- | :--- | :--- |
| `config.in_channels` | Số channel ảnh đầu vào | `1` |
| `down_channels[0]` | Số channel feature ban đầu | `32` |
| `kernel_size=3` | Bộ lọc nhìn vùng `3x3` | Nhìn vùng lân cận nhỏ |
| `padding=1` | Giữ nguyên kích thước ảnh | `32x32` vẫn là `32x32` |

Shape: `[64, 1, 32, 32]
        → [64, 32, 32, 32]`
        
Nghĩa là `conv0` biến ảnh xám một kênh thành feature map 32 kênh để U-Net bắt đầu xử lý.

**Giải thích phần đường xuống `self.downs`**
```python
        self.downs = nn.ModuleList([
            Block(
                down_channels[i],
                down_channels[i + 1],
                config.time_emb_dim,
                up=False,
            )
            for i in range(len(down_channels) - 1)
        ])
```

| Thành phần | Nó làm gì? | Trực giác |
| :--- | :--- | :--- |
| `nn.ModuleList(...)` | Lưu nhiều block PyTorch | Danh sách các tầng trong U-Net |
| `Block(..., up=False)` | Tạo block downsample | Giảm kích thước ảnh |
| `down_channels[i]` | Channel đầu vào của block | Channel hiện tại |
| `down_channels[i + 1]` | Channel đầu ra của block | Channel tầng tiếp theo |
| `range(len(down_channels) - 1)` | Tạo 4 block down | Đi từ 32 đến 512 channel |

Với `down_channels` = `(32, 64, 128, 256, 512)`. Ta tạo ra 4 block:

| Block | Input channel | Output channel | Spatial size |
| :--- | :--- | :--- | :--- |
| `downs[0]` | `32` | `64` | `32 -> 16` |
| `downs[1]` | `64` | `128` | `16 -> 8` |
| `downs[2]` | `128` | `256` | `8 -> 4` |
| `downs[3]` | `256` | `512` | `4 -> 2` |

**Giải thích phần đường lên `self.ups`**
```python
self.ups = nn.ModuleList([
    Block(
        up_channels[i],
        up_channels[i + 1],
        config.time_emb_dim,
        up=True,
    )
    for i in range(len(up_channels) - 1)
])
```

| Thành phần | Nó làm gì? | Trực giác |
| :--- | :--- | :--- |
| `Block(..., up=True)` | Tạo block upsample | Phóng to ảnh |
| `up_channels[i]` | Channel đầu vào logic của block | Channel hiện tại trước concat |
| `up_channels[i + 1]` | Channel đầu ra | Channel tầng tiếp theo |
| `ConvTranspose2d trong Block` | Tăng kích thước ảnh | Đi ngược từ bottleneck về ảnh lớn |

Với `up_channels` = `(512, 256, 128, 64, 32)`. Ta tạo ra 4 block:

| Block | Input channel logic | Output channel | Spatial size |
| :--- | :--- | :--- | :--- |
| `ups[0]` | `512` | `256` | `2 -> 4` |
| `ups[1]` | `256` | `128` | `4 -> 8` |
| `ups[2]` | `128` | `64` | `8 -> 16` |
| `ups[3]` | `64` | `32` | `16 -> 32` |

**Lưu ý:** Trong block up, `conv1` nhận `2 * in_ch` vì có skip connection.

**Giải thích phần `output`**
```python
self.output = nn.Conv2d(
    up_channels[-1],
    config.in_channels,
    kernel_size=1,
)
```

| Thành phần | Nó làm gì? |
| :--- | :--- |
| `up_channels[-1]` | Channel cuối của decoder, ở đây là `32` |
| `config.in_channels` | Số channel ảnh gốc, với MNIST là `1` |
| `kernel_size=1` | Convolution `1x1`, đổi số channel nhưng giữ nguyên kích thước ảnh |

Shape: `[64, 32, 32, 32]
        → [64, 1, 32, 32]`

Đầu ra này chính là noise dự đoán: $$\epsilon_\theta(x_t, t)$$

Nó phải có cùng shape với noise thật ($$\epsilon$$): `[64, 1, 32, 32]`

**Hàm `forward` của SimpleUNet**
```python
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Thêm kết nối tắt (skip connection)
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)
```

| Dòng code | Nó làm gì? | Trực giác | Shape ví dụ |
| :--- | :--- | :--- | :--- |
| `def forward(self, x, timestep):` | Nhận ảnh nhiễu và timestep | Model cần cả ảnh và thời gian | `x`: `[64, 1, 32, 32]`, `timestep`: `[64]` |
| `t = self.time_mlp(timestep)` | Biến timestep thành time embedding | Tạo đồng hồ thời gian cho U-Net | `[64, 256]` |
| `x = self.conv0(x)` | Đổi ảnh thành feature map | Từ ảnh 1 channel sang feature 32 channel | `[64, 32, 32, 32]` |
| `residual_inputs = []` | Tạo danh sách lưu skip connection | Lưu feature từ encoder để dùng ở decoder | `[]` |
| `for down in self.downs:` | Đi qua các block downsample | Đi xuống đáy U-Net | — |
| `x = down(x, t)` | Xử lý feature và giảm kích thước | Vừa học ảnh, vừa dùng timestep | Kích thước giảm dần |
| `residual_inputs.append(x)` | Lưu feature map | Chuẩn bị skip connection | Lưu nhiều tensor |
| `for up in self.ups:` | Đi qua các block upsample | Đi lên để phục hồi kích thước ảnh | — |
| `residual_x = residual_inputs.pop()` | Lấy feature đã lưu gần nhất | Lấy skip connection tương ứng | Tensor từ encoder |
| `torch.cat((x, residual_x), dim=1)` | Nối theo channel | Ghép thông tin decoder và encoder | Channel tăng gấp đôi |
| `x = up(x, t)` | Upsample feature | Phóng to ảnh dần | Kích thước tăng dần |
| `return self.output(x)` | Trả về noise dự đoán | Đưa output về 1 channel | `[64, 1, 32, 32]` |

**Skip connection là gì?**

Skip connection là đoạn này:
```python
residual_x = residual_inputs.pop()
x = torch.cat((x, residual_x), dim=1)
```

Nó lấy feature map đã lưu từ encoder, rồi nối với feature map hiện tại ở decoder.

**Vì sao cần skip connection?**
- Khi encoder giảm kích thước ảnh, một số chi tiết nhỏ có thể bị mất.
- Decoder cần dựng lại ảnh ở độ phân giải cao, nên nó cần thông tin chi tiết từ encoder.
- Skip connection giúp decoder không phải tự đoán lại toàn bộ chi tiết từ bottleneck.

**Vì sao dùng `torch.cat(..., dim=1)`?**

Tensor ảnh trong PyTorch thường có shape: `[batch, channel, height, width]`

Trong đó `dim=1` là chiều channel.

Khi ta viết: `x = torch.cat((x, residual_x), dim=1)` -> nghĩa là nối hai feature map theo chiều channel.

Ví dụ: 

       x:          [64, 256, 4, 4]

       residual_x: [64, 256, 4, 4] 
       
       sau cat:    [64, 512, 4, 4]

Ta không nối theo chiều height hoặc width, vì làm vậy sẽ làm méo cấu trúc ảnh. Ta nối theo channel để giữ nguyên kích thước không gian nhưng tăng lượng thông tin.
**Lưu ý:** Đầu ra của `SimpleUNet` phải có cùng shape với noise thật để tính loss.

**Vì sao dùng U-Net?**

U-Net là một kiến ​​trúc mã hóa-giải mã với một tính năng đặc biệt gọi là "kết nối bỏ qua" (skip connection)

U-Net rất hiệu quả trong bài toán image-to-image. Cả đầu vào và đầu ra đều có dạng ảnh.

Đầu vào | Đầu ra |
| :--- | :--- |
| Ảnh bị nhiễu $$(x_t)$$ | Nhiễu dự đoán $$(\epsilon_\theta(x_t,t))$$ |

Ví dụ với MNIST:

Input  $$x_t$$:          `[64, 1, 32, 32]`

Output predicted_noise:  `[64, 1, 32, 32]`

U-Net phù hợp vì nó có hai phần:

| Phần | Vai trò |
| :--- | :--- |
| Encoder / Down path	| Giảm kích thước ảnh để học ngữ cảnh tổng quát |
| Decoder / Up path | Phóng to lại để dự đoán chi tiết từng pixel |
| Skip connection	| Mang chi tiết từ encoder sang decoder |

Sau khi đã nắm được cấu trúc tổng thể của U-Net, bước tiếp theo là đi sâu vào thành phần cốt lõi của nó – các khối (block) – và tìm hiểu cách chúng tích hợp thông tin về thời gian vô cùng quan trọng.

### 2.7 Kiến trúc bên trong U-Net: Block và Time Embedding

Ở chương trước, chúng ta đã xem xét kiến ​​trúc tổng quan của mô hình `SimpleUNet`. Giờ đây, chúng ta sẽ đi sâu vào hai thành phần cốt lõi nhất của nó:

1. `SinusoidalPositionEmbeddings`: Cơ chế thông minh giúp mã hóa bước thời gian (timestep) t thành một vectơ hữu ích.

2. `Block`: Mô-đun chủ chốt có thể tái sử dụng, đảm nhiệm việc thực hiện các phép tích chập (convolution) và tích hợp thông tin về thời gian.


**Phần 1: Time Embedding — Cho mô hình biết timestep**

**Mã hóa thời gian bằng `SinusoidalPositionEmbeddings`**

Một thông tin cực kỳ quan trọng đối với mô hình của chúng ta là bước thời gian $$t$$. Mô hình cần biết liệu nó đang xử lý một hình ảnh có độ nhiễu thấp ($$t$$ nhỏ) hay độ nhiễu cao ($$t$$ lớn), vì chiến lược khử nhiễu cần được điều chỉnh cho phù hợp.

Ở Chương 2.3, ta đã chuẩn bị được ảnh gốc MNIST ($$x_0$$). Trong Diffusion Model, ta sẽ chọn một timestep ($$t$$), rồi thêm nhiễu vào ảnh gốc để tạo ra ảnh nhiễu ($$x_t$$). 
Công thức forward diffusion là:
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

Nhưng khi đưa ảnh nhiễu ($$x_t$$) vào mô hình, ta không thể chỉ đưa ảnh vào. Mô hình còn phải biết ảnh đó đang ở timestep nào.
| Bước thời gian (Timestep) | Mức độ nhiễu | Mô hình cần làm gì? |
| :--- | :--- | :--- |
| `(t = 10)` | Ít nhiễu | Khử nhiễu nhẹ |
| `(t = 30)` | Nhiễu trung bình | Khử nhiễu vừa phải |
| `(t = 299)` | Rất nhiều nhiễu | Khử nhiễu mạnh hơn |

Vì vậy, để thực hiện được nhiệm vụ này, mô hình cần phải nhận cả hai thông tin đầu vào cùng lúc:

```
  x_t, t
```
Trong đó:
- ($$x_t$$): ảnh đang bị nhiễu.
- ($$t$$): timestep hiện tại.

Mô hình sẽ học hàm:
$$\epsilon_\theta(x_t, t)$$

Nghĩa là mô hình nhìn ảnh bị nhiễu ($$x_t$$), đồng thời biết timestep ($$t$$), rồi dự đoán nhiễu trong ảnh.

Time Embedding giống như một chiếc đồng hồ báo cho mô hình biết:

"Nhiễu đã lan trong hình ảnh được bao lâu rồi?"

Nếu không có chiếc đồng hồ này, mô hình chỉ nhìn thấy hình ảnh bị nhiễu, nhưng không biết nó đang ở giai đoạn đầu, giữa hay cuối của quá trình diffusion.

Vấn đề: Chúng ta không thể chỉ đưa trực tiếp số nguyên $$t$$ (vd: 5, 250, 900) vào mạng nơ-ron. Đây chỉ là các giá trị vô hướng và không cung cấp đủ tín hiệu phong phú để mô hình diễn giải những khác biệt và mối quan hệ tinh tế giữa các bước thời gian.

Giải pháp: Chúng tôi sử dụng Sinusoidal Position Embeddings (Nhúng vị trí dạng sóng sin), một kỹ thuật được giới thiệu trong bài báo gốc "Attention Is All You Need" về Transformer. Kỹ thuật này ánh xạ một số nguyên $$t$$ duy nhất thành một vectơ đa chiều. Quá trình này tạo ra một "dấu vân tay" dạng sóng độc nhất cho mỗi bước thời gian. Nhờ đó, mô hình có thể dễ dàng học cách nhận diện các đặc điểm trong những dấu vân tay này để xác định mức độ nhiễu $$t$$.

Một timestep không còn là một số đơn lẻ nữa, mà trở thành một vector có nhiều tín hiệu thời gian khác nhau. Một số chiều trong vector thay đổi chậm. Một số chiều thay đổi nhanh. Nhờ đó, mạng có thể hiểu timestep ở nhiều mức độ khác nhau.

Ban đầu, timestep chỉ là một số: $$t$$ = 150

Sau time embedding, nó trở thành một vector:

    emb(t) = [0.71, -0.22, 0.91, ..., 0.33]

Ta không cần tự diễn giải từng số trong vector này. Điều quan trọng là mạng có thể dùng vector này để biết ảnh đang bị nhiễu ở mức nào.

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )
        return embeddings
```
| Thành phần | Vai trò chính | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `SinusoidalPositionEmbeddings` | Module biến timestep thành vector | Biến "số thời gian" thành tín hiệu dễ hiểu cho model |
| `dim` | Kích thước vector embedding | Timestep sẽ được mở rộng thành vector dài `dim` |
| `forward(self, time)` | Nhận timestep và trả về embedding | Khi đưa `t` vào, module trả về vector thời gian |
| `sin()` | Mã hóa timestep bằng sóng sin | Một kiểu tín hiệu thời gian |
| `cos()` | Mã hóa timestep bằng sóng cos | Một kiểu tín hiệu thời gian khác |
| `torch.cat(...)` | Ghép sin và cos lại | Tạo vector embedding hoàn chỉnh |

**Phần 2: Phân tích chi tiết Block (Viên gạch CNN của U-Net)**

Ở phần 1, ta đã biết cách biến timestep ($$t$$) thành một vector gọi là Time Embedding. Nhưng câu hỏi tiếp theo là:

```Làm sao đưa thông tin thời gian đó vào mô hình xử lý ảnh?```

Trong U-Net, ta không đưa time embedding vào một lần duy nhất ở đầu mô hình. Thay vào đó, ta đưa nó vào nhiều `Block` khác nhau trong quá trình xử lý ảnh.

Mỗi Block có nhiệm vụ:
- Nhận feature map của ảnh.
- Xử lý feature map bằng convolution.
- Nhận time embedding.
- Biến time embedding về đúng số channel.
- Cộng time embedding vào feature map.
- Tiếp tục xử lý feature map.
- Downsample hoặc upsample ảnh.

```python
class Block(nn.Module):
    # Simple Conv -> GroupNorm -> GELU block
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.act(h)
        h = self.norm1(h)

        time_emb = self.time_mlp(t)
        time_emb = self.act(time_emb)
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.conv2(h)
        h = self.act(h)
        h = self.norm2(h)

        return self.transform(h)
```

| Thành phần | Vai trò chính | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `Block` | Khối xử lý cơ bản của U-Net | Một “trạm xử lý ảnh” trong mô hình |
| `in_ch` | Số channel đầu vào | Feature map đi vào có bao nhiêu lớp thông tin |
| `out_ch` | Số channel đầu ra | Block sẽ tạo ra bao nhiêu lớp thông tin mới |
| `time_emb_dim` | Kích thước vector time embedding | Độ dài tín hiệu thời gian từ Chapter 3 |
| `up=False` | Block dùng cho đường đi xuống | Giảm kích thước ảnh |
| `up=True` | Block dùng cho đường đi lên | Phóng to kích thước ảnh |
| `time_mlp` | Biến time embedding về đúng số channel | Điều chỉnh “đồng hồ thời gian” cho vừa với feature map |
| `conv1`, `conv2` | Các lớp convolution | Trích xuất đặc trưng ảnh |
| `GroupNorm` | Chuẩn hóa feature map | Giúp training ổn định hơn |
| `GELU` | Hàm kích hoạt | Giúp mô hình học quan hệ phi tuyến |
| `transform` | Downsample hoặc upsample | Thay đổi kích thước không gian của ảnh |

**Vì sao cần `Block`?**

U-Net gồm nhiều tầng xử lý ảnh. Nếu ta viết từng tầng riêng lẻ, code sẽ rất dài và lặp lại. Vì vậy, ta tạo một class `Block` dùng lại nhiều lần. Một `Block` có thể dùng ở hai nơi:

| Vị trí trong U-Net | up | Nhiệm vụ |
| :--- | :--- | :--- |
| Encoder / Down | `False` | Giảm kích thước ảnh, tăng số channel |
| Decoder / Up | `True` | Giảm kích thước ảnh, tăng số channel |

**Giải thích phần `__init__`**

```python
def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = nn.Linear(time_emb_dim, out_ch)
```

| Dòng code | Nó làm gì? | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `def __init__(...)` | Hàm khởi tạo Block | Định nghĩa bên trong Block có những layer nào |
| `in_ch` | Số channel đầu vào | Feature map đi vào rộng bao nhiêu |
| `out_ch` | Số channel đầu ra | Feature map đi ra rộng bao nhiêu |
| `time_emb_dim` | Kích thước time embedding | Vector thời gian dài bao nhiêu |
| `up=False` | Quyết định block downsample hay upsample | Block đang ở đường xuống hay đường lên |
| `super().__init__()` | Khởi tạo lớp cha `nn.Module` | Bước bắt buộc khi viết module PyTorch |
| `nn.Linear(time_emb_dim, out_ch)` | Biến time embedding từ `time_emb_dim` sang `out_ch` | Làm cho vector thời gian có cùng số channel với feature map |

**Giải thích `Block` ở đường xuống (down): `up=False`**

```python
else:
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
```

| Thành phần | Nó làm gì? | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `self.conv1` | Đổi feature map từ `in_ch` sang `out_ch` | Trích xuất đặc trưng ảnh |
| `kernel_size=3` | Dùng bộ lọc `3x3` | Nhìn vùng lân cận nhỏ quanh mỗi pixel |
| `padding=1` | Thêm viền ảo quanh ảnh | Giữ nguyên kích thước không gian sau convolution |
| `self.transform` | Convolution với `stride=2` | Giảm chiều cao và chiều rộng ảnh đi một nửa |
| `kernel_size=4, stride=2, padding=1` | Công thức downsample phổ biến | Giúp `32 -> 16 -> 8 -> 4 -> 2` |

Ví dụ shape khi đi xuống: `[64, 32, 32, 32] → [64, 64, 16, 16]`

| Thành phần | Trước Block | Sau Block |
| :--- | :--- | :--- |
| Batch | `64` | `64` |
| Channel | `32` | `64` |
| Height | `32` | `16` |
| Width | `32` | `16` |

Trong `Encoder`, ảnh nhỏ dần nhưng channel tăng dần. Điều này giúp mô hình học đặc trưng tổng quát hơn.

**Giải thích `Block` ở đường lên (up): `up=True`**

```python
if up:
    self.conv1 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=3, padding=1)
    self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
```

| Thành phần | Nó làm gì? | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `up=True` | Block dùng ở decoder | Đường đi lên trong U-Net |
| `2 * in_ch` | Vì có skip connection | Feature hiện tại được nối với feature từ encoder |
| `nn.Conv2d(2 * in_ch, out_ch, ...)` | Giảm số channel sau khi concat | Trộn thông tin từ decoder và encoder |
| `ConvTranspose2d` | Upsampling bằng convolution ngược | Phóng to ảnh lên gấp đôi |
| `stride=2` | Tăng chiều cao và chiều rộng | Ví dụ `8 -> 16` |

**Vì sao là `2 * in_ch`?**

Trong U-Net, ở đường lên (up) ta nối feature hiện tại với feature được lưu từ đường xuống (down): `x = torch.cat((x, residual_x), dim=1)`

Việc nối theo `dim=1` nghĩa là nối theo channel.

Ví dụ:

x hiện tại:      `[64, 128, 8, 8]`
residual_x:      `[64, 128, 8, 8]`
sau torch.cat:   `[64, 256, 8, 8]`

Số channel tăng gấp đôi, nên `conv1` ở block up phải nhận `2 * in_ch`

**Các layer chung trong `Block`**

```python
self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
self.norm1 = nn.GroupNorm(8, out_ch)
self.norm2 = nn.GroupNorm(8, out_ch)
self.act = nn.GELU()
```

| Thành phần | Nó làm gì? | Trực giác dễ hiểu |
| :--- | :--- | :--- |
| `conv2` | Tiếp tục xử lý feature map | Sau khi thêm time embedding, ta cho model học tiếp |
| `GroupNorm(8, out_ch)` | Chia channel thành 8 nhóm để chuẩn hóa | Giúp mô hình học ổn định hơn |
| `GELU()` | Hàm kích hoạt phi tuyến | Giúp mô hình học quan hệ phức tạp |
| `padding=1` | Giữ nguyên kích thước sau conv `3x3` | Không làm ảnh bị nhỏ sau mỗi convolution thường |

**Vì sao dùng `GroupNorm` thay vì `BatchNorm`?**

Trong Diffusion Model, batch size đôi khi không lớn, nhất là khi train ảnh lớn hoặc model lớn.

`BatchNorm` phụ thuộc nhiều vào thống kê của batch. Nếu batch nhỏ, thống kê này có thể không ổn định.

`GroupNorm` chuẩn hóa theo nhóm channel trong từng ảnh, nên thường ổn định hơn với batch nhỏ.

| Normalization | Phụ thuộc batch size nhiều không? | Phù hợp Diffusion không? |
| :--- | :--- | :--- |
| `BatchNorm` | Có | Ít phù hợp hơn khi batch nhỏ |
| `GroupNorm` | Ít hơn | Phù hợp hơn |

**Giải thích hàm `forward`**

```python
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.act(h)
        h = self.norm1(h)

        time_emb = self.time_mlp(t)
        time_emb = self.act(time_emb)
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.conv2(h)
        h = self.act(h)
        h = self.norm2(h)

        return self.transform(h)
```

| Dòng code | Nó làm gì? | Trực giác dễ hiểu | Shape ví dụ |
| :--- | :--- | :--- | :--- |
| `def forward(self, x, t):` | Nhận feature map `x` và time embedding `t` | Block cần cả ảnh và thời gian | `x: [64, 32, 32, 32]`, `t: [64, 256]` |
| `h = self.conv1(x)` | Convolution đầu tiên | Trích xuất đặc trưng ảnh | `[64, 64, 32, 32]` |
| `h = self.act(h)` | Áp dụng GELU | Thêm khả năng học phi tuyến | `[64, 64, 32, 32]` |
| `h = self.norm1(h)` | Chuẩn hóa feature map | Giúp training ổn định | `[64, 64, 32, 32]` |
| `time_emb = self.time_mlp(t)` | Đổi time embedding về `out_ch` | Làm thời gian khớp với số channel | `[64, 64]` |
| `time_emb = self.act(time_emb)` | Áp dụng GELU cho time embedding | Cho tín hiệu thời gian linh hoạt hơn | `[64, 64]` |
| `time_emb = time_emb[:, :, None, None]` | Thêm 2 chiều không gian | Chuẩn bị cộng vào feature map | `[64, 64, 1, 1]` |
| `h = h + time_emb` | Cộng thời gian vào feature map | Mỗi pixel nhận thêm thông tin timestep | `[64, 64, 32, 32]` |
| `h = self.conv2(h)` | Convolution lần hai | Xử lý ảnh sau khi đã biết thời gian | `[64, 64, 32, 32]` |
| `h = self.act(h)` | GELU | Tăng khả năng biểu diễn | `[64, 64, 32, 32]` |
| `h = self.norm2(h)` | GroupNorm lần hai | Ổn định feature map | `[64, 64, 32, 32]` |
| `return self.transform(h)` | Downsample hoặc upsample | Đổi kích thước ảnh | Down: `[64, 64, 16, 16]` |

**Vì sao cần `time_emb[:, :, None, None]`?**

Trước khi reshape, time embedding có shape: `[batch_size, channels]`

Ví dụ: `[64, 64]`

Nhưng feature map có shape: `[batch_size, channels, height, width]`

Ví dụ: `[64, 64, 32, 32]` 

Ta không thể cộng trực tiếp: `[64, 64]` + `[64, 64, 32, 32]`

Vì vậy ta đổi time embedding thành: `[64, 64, 1, 1]`

bằng dòng: `time_emb = time_emb[:, :, None, None]`

Sau đó PyTorch tự broadcast: `[64, 64, 1, 1]` → `[64, 64, 32, 32]`

Nghĩa là cùng một tín hiệu thời gian được cộng vào mọi vị trí pixel trong feature map

`Block là nơi feature ảnh và thông tin thời gian gặp nhau. Nhờ đó, U-Net không chỉ nhìn thấy ảnh nhiễu, mà còn biết ảnh đó đang nhiễu ở timestep nào.`











