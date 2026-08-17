# 🧠 Mạng Nơ-ron Tích Chập (Convolutional Neural Networks - CNN)

Chào mừng bạn đến với tài liệu hướng dẫn về **CNN và các kiến trúc biến thể kinh điển** trong thị giác máy tính (Computer Vision). Tài liệu này được tổng hợp ngắn gọn, trực quan nhằm giúp người mới bắt đầu học AI nhanh chóng nắm bắt bản chất cốt lõi của từng mô hình qua các thời kỳ.

---

## 📌 Nội Dung Tổng Quan

Tài liệu bao gồm các tệp PDF tóm tắt chi tiết cấu trúc, thông số và các đóng góp quan trọng của từng mô hình:

* **`1998_LeNet5.pdf`**: Nền móng của mạng tích chập (dành cho bài toán nhận dạng chữ số viết tay MNIST).
* **`2012_AlexNet.pdf`**: Bước ngoặt lịch sử khởi đầu kỷ nguyên Deep Learning và sự bùng nổ của GPU.
* **`2014_VGG16.pdf`**: Tối ưu hóa kiến trúc bằng cách dùng bộ lọc nhỏ $3 \times 3$ và tăng độ sâu mạng.
* **`2015_ResNet.pdf`**: Giải pháp kết nối tắt (Skip Connection / Residual Learning) giúp giải quyết triệt để vấn đề tiêu biến đạo hàm (Vanishing Gradient) khi huấn luyện mạng siêu sâu.

---

## 🚀 Tóm Tắt Nhanh Các Mạng CNN Kinh Điển

| Mô hình | Năm | Số lớp có trọng số | Điểm nổi bật chính |
| :--- | :---: | :---: | :--- |
| **LeNet-5** | 1998 | 7 lớp | Sử dụng kiến trúc cơ bản: *Convolution $\rightarrow$ Subsampling $\rightarrow$ Full Connection*. |
| **AlexNet** | 2012 | 8 lớp | Dùng hàm kích hoạt **ReLU**, kỹ thuật **Dropout**, **Max Pooling** và huấn luyện trên GPU. |
| **VGG16** | 2014 | 16 lớp | Chuẩn hóa bộ lọc kích thước nhỏ $3 \times 3$ xếp chồng lên nhau, giúp mạng sâu nhưng nhất quán. |
| **ResNet** | 2015 | 18, 34, 50, 101, 152+ lớp | Giới thiệu **Residual Block** & **Skip Connection**, mở đường cho việc huấn luyện các mạng cực sâu. |

---

## 🎯 Lộ Trình Học Tập Đề Xuất

1. **Bước 1:** Đọc file `1998_LeNet5.pdf` để hiểu khái niệm tích chập (Convolution), bước nhảy (Stride), đệm (Padding) và gộp (Pooling).
2. **Bước 2:** Đọc file `2012_AlexNet.pdf` để nắm được lý do tại sao Deep Learning bùng nổ và các kỹ thuật chống quá khớp (Overfitting).
3. **Bước 3:** Đọc file `2014_VGG16.pdf` để thấy cách thiết kế mạng sâu hơn và ứng dụng cho bài toán Transfer Learning.
4. **Bước 4:** Đọc file `2015_ResNet.pdf` để hiểu cơ chế học thặng dư (Residual Learning) - tiêu chuẩn nền tảng cho nhiều mạng Deep Learning hiện đại.

---

## 🛠 Luyện Tập Thực Hành

Sau khi đọc xong lý thuyết trong các file PDF, bạn nên thực hành cài đặt lại các kiến trúc này bằng các framework như **PyTorch** hoặc **TensorFlow/Keras**:

```python
# Ví dụ dựng khối Convolution cơ bản trong PyTorch
import torch.nn as nn

conv_layer = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
