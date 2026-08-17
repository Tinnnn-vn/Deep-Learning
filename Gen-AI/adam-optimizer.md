```
Bạn biết thuật toán Gradient Descent. Bạn biết công thức: trọng số mới = trọng số cũ - tốc độ học * độ dốc. Nó đơn giản, hiệu quả. Nhưng đó không phải là thứ tạo nên sức mạnh của AI hiện đại.

Trong mọi mô hình tiên tiến, trong mọi kịch bản huấn luyện hiệu năng cao, bạn đều thấy cùng một cái tên: Adam. Bạn được khuyên chỉ cần sử dụng nó. Nó là mặc định. Nó "tốt hơn".

Nhưng tại sao?

Dưới đây là toàn bộ bí mật: Adam không phải là một ý tưởng phức tạp duy nhất. Nó thực chất là sự kết hợp của ba ý tưởng đơn giản, được ghép lại với nhau để giải quyết ba vấn đề cụ thể.
```
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

## 2. Ý tưởng thứ nhất: nhớ hướng đi bằng Momentum

Hãy tưởng tượng một quả bóng lăn xuống dốc. Nó không quên chuyển động trước đó mà có **quán tính**. Nếu liên tục được đẩy cùng hướng, nó sẽ đi nhanh hơn. Nếu lực đổi hướng liên tục, các lực sẽ phần nào triệt tiêu nhau.

Adam lưu một đại lượng gọi là **moment bậc nhất**:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

Đọc công thức như sau:

> Hướng hiện tại = một phần hướng đã nhớ + một phần gradient mới.

Các ký hiệu:

- $g_t$: gradient tại bước `t`;
- $m_t$: hướng trung bình được ghi nhớ đến bước `t`;
- $\beta_1$: mức độ ghi nhớ, thường là `0.9`.

Nếu $\beta_1=0.9$, Adam giữ `90%` thông tin cũ và nhận `10%` thông tin mới. Nhờ vậy, đường đi bớt rung lắc và ổn định hơn.

> $m$ trả lời câu hỏi: **“Ta thường nên đi theo hướng nào?”**

---

## 3. Ý tưởng thứ hai: đo độ lớn của gradient

Không phải tham số nào cũng có gradient giống nhau:

- gradient lớn: tham số rất nhạy, bước cập nhật cần thận trọng;
- gradient nhỏ: tham số ít nhạy, có thể cần bước tương đối mạnh hơn.

Adam theo dõi trung bình của **bình phương gradient**:

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Vì được bình phương nên mọi giá trị đều không âm. Gradient càng lớn thì $g_t^2$ càng lớn.

- $v_t$: độ lớn gradient được ghi nhớ;
- $\beta_2$: mức độ ghi nhớ, thường là `0.999`.

Adam dùng $v$ để điều chỉnh bước học riêng cho từng tham số:

- $v$ lớn → mẫu số lớn → bước cập nhật nhỏ;
- $v$ nhỏ → mẫu số nhỏ → bước cập nhật tương đối lớn.

> $v$ trả lời câu hỏi: **“Ở hướng này, ta nên bước mạnh hay nhẹ?”**

### Vì sao không cộng tất cả gradient bình phương?

AdaGrad, một thuật toán ra đời trước Adam, cộng toàn bộ gradient bình phương từ đầu buổi học. Tổng này chỉ có thể tăng, khiến learning rate ngày càng nhỏ và có thể gần như ngừng học.

Adam dùng **trung bình trượt có trọng số mũ** (*EWMA*). Thông tin cũ giảm ảnh hưởng dần, vì thế thuật toán vẫn thích nghi được khi tình hình thay đổi.

---

## 4. Tại sao cần sửa sai lệch ban đầu?

Lúc bắt đầu, Adam đặt:

```text
m = 0
v = 0
```

Do khởi đầu từ số `0`, các giá trị trung bình trong vài bước đầu bị nhỏ hơn thực tế. Hiện tượng này gọi là **bias về 0**.

Adam sửa lại bằng:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Ở đây, dấu mũ `^` trên $\hat m$ và $\hat v$ chỉ giá trị **đã được hiệu chỉnh**, không phải phép lũy thừa.

Khi `t` còn nhỏ, phép sửa có tác dụng rõ rệt. Khi `t` lớn, $\beta^t$ tiến gần `0`, mẫu số tiến gần `1` và việc hiệu chỉnh gần như không còn ảnh hưởng.

---
