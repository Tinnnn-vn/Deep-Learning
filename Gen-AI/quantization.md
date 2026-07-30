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


