# Quantization: Từ FP32 đến INT8 và INT4

Một bài hướng dẫn đi từ trực giác, công thức đến mã Python để hiểu cách mô hình AI được nén xuống 8-bit hoặc 4-bit.

## I. Quantization là gì?

Quantization (lượng tử hóa) là kỹ thuật biểu diễn các con số có độ chính xác cao bằng một tập số nhỏ hơn.

Trong mô hình AI, trọng số thường được lưu bằng:

`FP32`: mỗi trọng số dùng 32 bit.

`FP16`: mỗi trọng số dùng 16 bit.

`INT8`: mỗi trọng số dùng 8 bit.

`INT4`: mỗi trọng số dùng 4 bit.

Hãy tưởng tượng bạn có một chiếc thước đo rất chi tiết:

0.000, 0.001, 0.002, 0.003, ...

Chiếc thước này đo chính xác, nhưng cần rất nhiều vạch.

Quantization giống như đổi sang chiếc thước ít vạch hơn:

0.0, 0.1, 0.2, 0.3, ...

Ta tiết kiệm không gian, nhưng phải chấp nhận rằng một số giá trị sẽ được làm tròn.

Ví dụ:
```
Giá trị thật:       1.23
Giá trị sau nén:    1.20
Sai số:             0.03
```
Điều quan trọng không phải là loại bỏ hoàn toàn sai số mà là: `Giảm bộ nhớ thật nhiều nhưng giữ sai số đủ nhỏ để mô hình vẫn hoạt động tốt.`

## II. Tại sao cần Quantization?

Một mô hình ngôn ngữ lớn (LLM) có thể chứa hàng tỷ trọng số. Mỗi trọng số chỉ là một con số, nhưng khi số lượng lên đến hàng tỷ, bộ nhớ cần thiết trở nên rất lớn.

**Ví dụ với mô hình 7 tỷ tham số**

Cách tính gần đúng: `Bộ nhớ = số tham số × số byte cho mỗi tham số`

| Định dạng | Bit/trọng số | Byte/trọng số | Bộ nhớ lý thuyết cho 7B tham số |
| :--- | :--- | :--- | :--- |
| FP32 | 32 | 4 | khoảng 28 GB |
| FP16 | 16 | 2 | khoảng 14 GB |
| INT8 | 8 | 1 | khoảng 7 GB |
| INT4 | 4 | 0.5 | khoảng 3.5 GB |

Các con số trên chỉ tính phần trọng số. Khi chạy thực tế, chương trình còn cần bộ nhớ cho activation, KV cache, buffer và các metadata khác.

Quantization giúp giải quyết hai vấn đề lớn:

**1. Giảm dung lượng mô hình**

Mô hình nhỏ hơn sẽ:

- dễ lưu trên ổ cứng hơn

- dễ tải xuống hơn

- có thể chạy trên GPU ít VRAM hơn

- có cơ hội chạy trên laptop hoặc thiết bị biên

**2. Giảm lượng dữ liệu phải chuyển trong GPU**

Trong quá trình suy luận (inference), GPU liên tục đọc trọng số từ VRAM để thực hiện phép nhân ma trận. Nếu trọng số nhỏ hơn, lượng dữ liệu cần truyền cũng nhỏ hơn.

Trong nhiều trường hợp, tốc độ suy luận không chỉ bị giới hạn bởi khả năng tính toán. Nó còn bị giới hạn bởi tốc độ đưa trọng số từ bộ nhớ đến nhân xử lý.

## III. FP32 lưu một con số như thế nào?

Một số `FP32` sử dụng 32 bit, được chia thành ba phần:

| Thành phần | Số bit | Vai trò |
| :--- | :--- | :--- |
| Sign | 1 | Cho biết số âm hay dương |
| Exponent | 8 | Điều khiển độ lớn của số |
| Mantissa/Fraction | 23 | Giữ phần chi tiết của số |

Cách biểu diễn này giúp FP32 chứa được: 
- số rất nhỏ;
- số rất lớn;
- số thập phân khá chính xác.

Ví dụ, số 3.5 trong hệ nhị phân là: `3.5 = 11.1₂`

Chuẩn hóa theo dạng khoa học nhị phân: `1.11 × 2¹`

Từ đó FP32 lưu:

[sign]   [exponent]   [mantissa]

[  0 ]   [10000000]   [11000000000000000000000]

Bạn không cần ghi nhớ cách chuyển đổi từng bit để hiểu quantization. Điều cần nhớ là:

`FP32 rất linh hoạt và chính xác, nhưng mỗi số tốn 32 bit.`

Trong khi đó, `INT8` chỉ là một số nguyên nằm trong khoảng: `-128 đến 127`. INT8 không có exponent và mantissa. Vì vậy, muốn chuyển FP32 sang INT8, ta cần tạo một phép ánh xạ.

## IV. Đổi từ số thực sang số nguyên

Giả sử một nhóm trọng số nằm trong khoảng: `-3.5 đến 3.5`

Trong khi INT8 có 256 giá trị có thể sử dụng: `-128, -127, ..., 0, ..., 126, 127`

Ta có thể tưởng tượng 256 số nguyên này là 256 chiếc hộp. Mỗi số thực sẽ được đặt vào chiếc hộp gần nó nhất.
```
Thế giới số thực:  -3.5 ------------------------------- 3.5

                       ↓ chia thành các bước nhỏ ↓

Thế giới INT8:     -128 ------------------------------- 127
```

Vì số hộp là hữu hạn, nhiều số thực gần nhau có thể rơi vào cùng một hộp. Ví dụ:
```text
0.51  ┐
      ├─> cùng được biểu diễn bằng số nguyên 2
0.58  ┘
```

Đây chính là nguồn gốc của quantization error, tức sai số lượng tử hóa.

## V. Scale và Zero-point thực sự có ý nghĩa gì?

Quantization thường được điều khiển bởi hai tham số:

- Scale, ký hiệu S.
- Zero-point, ký hiệu Z.

**1. Scale là kích thước của một bước**

Scale trả lời câu hỏi: `Khi số nguyên tăng thêm 1, giá trị trong thế giới số thực thay đổi bao nhiêu?`

Giả sử miền số thực là: `[-3.5, 3.5]`

Có hai cách nhìn thường gặp:

Affine quantization tổng quát dùng cả độ rộng miền float và miền integer:
```
S = (x_max - x_min) / (q_max - q_min)
S = 7.0 / 255 ≈ 0.02745
```

Symmetric quantization buộc tâm của hai miền trùng tại số 0 và thường chọn:
```
S = max(|x|) / 127
S = 3.5 / 127 ≈ 0.02756
```

Hai kết quả khá gần nhau trong ví dụ đối xứng này, nhưng chúng đến từ hai cách thiết lập miền khác nhau. Phần mã phía dưới sử dụng cách symmetric với `S = abs_max / 127`.

Nghĩa là một bước trong INT8 tương ứng khoảng `0.02756` ở miền số thực. Bạn có thể hình dung:
```
INT8 tăng 1 đơn vị
        ↓
FP32 tăng khoảng 0.02745
```

**2. Zero-point là vị trí của số 0**

Zero-point trả lời câu hỏi: `Giá trị float 0.0 sẽ được biểu diễn bằng số nguyên nào?`

Nếu miền dữ liệu đối xứng quanh 0, ta thường chọn: `Z = 0`. Cách này được gọi là symmetric quantization.

Ví dụ:
```
-3.5 -------- 0.0 -------- 3.5
-127 --------  0  -------- 127
```
Đối với trọng số mô hình, symmetric quantization rất phổ biến vì:
- công thức đơn giản;
- không cần lưu zero-point khác 0;
- phép tính trên phần cứng dễ tối ưu hơn.

## VI. Công thức quantization và dequantization

**1. Quantization: float → integer**

Công thức tổng quát:

$$q = \text{clamp}\left(\text{round}\left(\frac{x}{S} + Z\right), q_{\min}, q_{\max}\right)$$

Trong đó:

- x: giá trị float ban đầu
- S: scale
- Z: zero-point
- round: làm tròn về số nguyên gần nhất
- clamp: chặn kết quả để không vượt khỏi miền số nguyên
- q: giá trị sau quantization

Với symmetric quantization, `Z = 0`, công thức rút gọn thành:

$$q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), q_{\min}, q_{\max}\right)$$

**2. Dequantization: integer → float xấp xỉ**

Công thức: $$\hat{x} = S(q - Z)$$

Ký hiệu x̂ không phải là số ban đầu chính xác. Nó là giá trị xấp xỉ được dựng lại.
```
x   = giá trị ban đầu
x̂  = giá trị gần đúng sau khi nén rồi giải nén
```

Sai số: $$\text{error} = x - \hat{x}$$

*Vì sao phải có clamp?*

Giả sử một phép tính cho ra `q = 150`, nhưng INT8 chỉ chứa tối đa `127`.

Khi đó:

clamp(150, -128, 127) = 127

Nếu không chặn, giá trị có thể bị tràn số và cho kết quả sai hoàn toàn.













