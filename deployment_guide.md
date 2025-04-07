# Hướng dẫn triển khai ứng dụng PINNs tính toán mặt cắt kinh tế đập bê tông trọng lực

## Giới thiệu

Tài liệu này hướng dẫn cách triển khai ứng dụng tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng mô hình Physics-Informed Neural Networks (PINNs) lên Streamlit Cloud. Ứng dụng đã được cải tiến để khắc phục các lỗi và tối ưu hóa hiệu suất.

## Cải tiến chính

1. **Tối ưu hóa PINNs**:
   - Cải tiến hàm mất mát để đáp ứng chính xác 3 điều kiện: K=Kc, σ≈0, và tối thiểu hóa A
   - Thêm cơ chế hội tụ sớm để giảm thời gian tính toán
   - Thêm lịch trình học tập (learning rate scheduler) để cải thiện hội tụ

2. **Sửa lỗi cơ sở dữ liệu**:
   - Triển khai kết nối SQLite an toàn cho thread
   - Sử dụng thread-local storage và context manager để quản lý kết nối

3. **Cải thiện hiển thị đồ họa**:
   - Thêm hàm plot_loss_curve để hiển thị biểu đồ hàm mất mát
   - Cải thiện hiển thị biểu đồ mặt cắt đập

4. **Tương thích với Streamlit Cloud**:
   - Cập nhật yêu cầu PyTorch để tương thích với Python 3.12.9

## Yêu cầu

- Tài khoản GitHub
- Tài khoản Streamlit Cloud

## Các bước triển khai

### 1. Tạo repository trên GitHub

1. Đăng nhập vào GitHub
2. Nhấp vào nút "+" ở góc trên bên phải và chọn "New repository"
3. Điền thông tin:
   - **Repository name**: `dam-optimizer-pinns`
   - **Description** (tùy chọn): `Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs`
   - **Visibility**: Chọn "Public"
   - **Initialize this repository with**: Không chọn gì
4. Nhấp vào "Create repository"

### 2. Tải mã nguồn lên GitHub

1. Tải xuống các file đã cải tiến:
   - `app.py`
   - `pinns_optimizer.py`
   - `database.py`
   - `requirements.txt`

2. Tạo thư mục cục bộ và sao chép các file vào:
   ```bash
   mkdir -p dam-optimizer-pinns/data
   cp app.py pinns_optimizer.py database.py requirements.txt dam-optimizer-pinns/
   ```

3. Khởi tạo Git repository và đẩy mã nguồn lên GitHub:
   ```bash
   cd dam-optimizer-pinns
   git init
   git add .
   git commit -m "Initial commit with improved PINNs implementation"
   git branch -M main
   git remote add origin https://github.com/username_của_bạn/dam-optimizer-pinns.git
   git push -u origin main
   ```

   Thay `username_của_bạn` bằng tên người dùng GitHub của bạn.

### 3. Triển khai lên Streamlit Cloud

1. Đăng nhập vào [Streamlit Cloud](https://streamlit.io/cloud)
2. Nhấp vào "New app"
3. Trong phần "Repository", chọn repository `dam-optimizer-pinns` của bạn
4. Trong phần "Branch", chọn `main`
5. Trong phần "Main file path", nhập `app.py`
6. Nhấp vào "Deploy"

## Cấu trúc dự án

```
dam-optimizer-pinns/
├── app.py                  # Ứng dụng Streamlit chính
├── pinns_optimizer.py      # Mô-đun tối ưu hóa PINNs
├── database.py             # Mô-đun cơ sở dữ liệu thread-safe
├── requirements.txt        # Các thư viện cần thiết
└── data/                   # Thư mục lưu trữ cơ sở dữ liệu
```

## Giải thích chi tiết về cải tiến

### 1. Tối ưu hóa PINNs

Phiên bản cải tiến sử dụng một hàm mất mát tốt hơn để đáp ứng chính xác 3 điều kiện:

```python
def loss_function(sigma, K, A, Kc, alpha):
    # Điều kiện ổn định trượt: K=Kc (không phải K>=Kc)
    penalty_K = (K - Kc)**2
    
    # Điều kiện ứng suất mép thượng lưu: σ≈0 (không có ứng suất kéo)
    penalty_sigma = torch.where(sigma > 0, 
                               100 * sigma**2,  # Phạt nặng nếu có ứng suất kéo
                               (sigma - 0)**2)  # Khuyến khích sigma tiến gần đến 0
    
    # Tối thiểu hóa diện tích mặt cắt A
    objective = alpha * A
    
    return penalty_K.mean() + penalty_sigma.mean() + objective.mean()
```

Cơ chế hội tụ sớm giúp giảm thời gian tính toán:

```python
# Kiểm tra điều kiện hội tụ
if current_loss < best_loss:
    best_loss = current_loss
    best_params = (n.detach().clone(), m.detach().clone(), xi.detach().clone())
    patience_counter = 0
else:
    patience_counter += 1

# Kiểm tra điều kiện dừng sớm
if patience_counter >= patience:
    print(f"Dừng sớm tại epoch {epoch} do không cải thiện sau {patience} vòng lặp")
    break
```

### 2. Sửa lỗi cơ sở dữ liệu

Sử dụng thread-local storage để đảm bảo an toàn cho thread:

```python
thread_local = threading.local()

@contextmanager
def get_connection(self):
    # Kiểm tra xem thread hiện tại đã có kết nối chưa
    if not hasattr(thread_local, 'connection'):
        # Tạo kết nối mới cho thread hiện tại
        thread_local.connection = sqlite3.connect(self.db_path)
    
    try:
        # Trả về kết nối cho context
        yield thread_local.connection
    finally:
        # Không đóng kết nối ở đây để tái sử dụng trong cùng một thread
        pass
```

### 3. Cải thiện hiển thị đồ họa

Thêm hàm plot_loss_curve để hiển thị biểu đồ hàm mất mát:

```python
def plot_loss_curve(loss_history):
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Thêm đường biểu diễn hàm mất mát
    fig.add_trace(go.Scatter(
        x=np.arange(len(loss_history)),
        y=loss_history,
        mode='lines',
        name='Hàm mất mát',
        line=dict(color='#0066cc', width=2)
    ))
    
    # Cấu hình chung
    fig.update_layout(
        title='Biểu đồ hàm mất mát theo vòng lặp',
        xaxis_title='Vòng lặp',
        yaxis_title='Giá trị hàm mất mát',
        width=850,
        height=500,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig
```

## Xử lý sự cố

### Nếu gặp lỗi khi triển khai

1. Kiểm tra logs lỗi bằng cách nhấp vào "Manage app" và xem phần "Logs"
2. Đảm bảo rằng bạn đã đẩy tất cả các file lên GitHub, bao gồm `app.py`, `pinns_optimizer.py`, `database.py` và `requirements.txt`
3. Đảm bảo file `requirements.txt` có nội dung:
   ```
   -f https://download.pytorch.org/whl/cpu/torch_stable.html
   torch>=2.2.0
   streamlit>=1.20.0
   numpy>=1.26.0
   pandas>=2.0.0
   matplotlib>=3.7.0
   plotly>=5.15.0
   openpyxl>=3.0.0
   ```
4. Nếu vẫn gặp vấn đề, hãy thử các bước sau:
   - Kiểm tra xem repository của bạn có công khai (public) không
   - Đảm bảo rằng bạn đã cấp quyền cho Streamlit Cloud truy cập vào repository của bạn
   - Thử tạo lại ứng dụng

## Kết luận

Ứng dụng đã được cải tiến để khắc phục các lỗi và tối ưu hóa hiệu suất. Các cải tiến chính bao gồm:

1. Tối ưu hóa PINNs với hàm mất mát tốt hơn và cơ chế hội tụ sớm
2. Sửa lỗi cơ sở dữ liệu với kết nối SQLite an toàn cho thread
3. Cải thiện hiển thị đồ họa với biểu đồ hàm mất mát
4. Tương thích với Streamlit Cloud thông qua cập nhật yêu cầu PyTorch

Ứng dụng giờ đây có thể tính toán mặt cắt kinh tế đập bê tông thỏa mãn chính xác 3 điều kiện: K=Kc, σ≈0, và tối thiểu hóa A.
