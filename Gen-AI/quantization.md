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



