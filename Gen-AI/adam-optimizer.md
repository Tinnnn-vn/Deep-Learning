Bạn biết thuật toán Gradient Descent. Bạn biết công thức: trọng số mới = trọng số cũ - tốc độ học * độ dốc. Nó đơn giản, hiệu quả. Nhưng đó không phải là thứ tạo nên sức mạnh của AI hiện đại.

Trong mọi mô hình tiên tiến, trong mọi kịch bản huấn luyện hiệu năng cao, bạn đều thấy cùng một cái tên: Adam. Bạn được khuyên chỉ cần sử dụng nó. Nó là mặc định. Nó "tốt hơn".

Nhưng tại sao?

# Adam Optimizer – Giải thích dễ hiểu từ con số 0

Adam là một thuật toán giúp mô hình AI **điều chỉnh các tham số trong lúc học**. Có thể hình dung nó như một người lái xe thông minh:

- nhớ hướng nào vừa đi là đúng;
- giảm tốc ở nơi dốc và nguy hiểm;
- tăng tốc ở nơi bằng phẳng;
- tự chọn bước đi phù hợp cho từng tham số.

Tên **Adam** xuất phát từ cụm *Adaptive Moment Estimation*, có thể hiểu gần đúng là **ước lượng chuyển động để tự điều chỉnh bước học**.

---

## 1. Mô hình AI “học” như thế nào?

Trong quá trình huấn luyện, mô hình có các tham số như `w`. Ta muốn thay đổi chúng để hàm mất mát (*loss*) ngày càng nhỏ.

Gradient Descent cập nhật tham số theo công thức:

$$
w_{mới}=w_{cũ}-\eta g
$$

Trong đó:

- $w$: tham số cần học;
- $g$: gradient, cho biết độ dốc và hướng làm loss tăng nhanh nhất;
- $-g$: hướng nên đi để giảm loss;
- $\eta$: learning rate, tức độ dài của mỗi bước.

### Ví dụ ngắn

Giả sử:

$$
f(w)=w^2, \qquad g=f'(w)=2w
$$

Với $w=10$ và $\eta=0.1$:

$$
w_{mới}=10-0.1\times20=8
$$

Ta đã tiến từ `10` về gần điểm nhỏ nhất `0`.

Gradient Descent đơn giản, nhưng nó có ba hạn chế:

1. Không nhớ những bước trước nên có thể tiến chậm hoặc đi zíc zắc.
2. Dùng một learning rate chung dù mỗi tham số có độ nhạy khác nhau.
3. Nếu learning rate quá lớn, quá trình học có thể vượt qua điểm tốt và mất ổn định.

Adam được tạo ra để xử lý những vấn đề này.

---
