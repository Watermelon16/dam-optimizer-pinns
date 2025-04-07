# Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs

## Giới thiệu

Đây là ứng dụng tính toán tối ưu mặt cắt kinh tế của đập bê tông trọng lực sử dụng mô hình Physics-Informed Neural Networks (PINNs). Ứng dụng này kết hợp kiến thức vật lý về cơ học đập với sức mạnh của học sâu để tìm ra mặt cắt tối ưu nhất, đảm bảo các điều kiện ổn định và an toàn.

## Tính năng

- **Tính toán tối ưu sử dụng PINNs**: Áp dụng mô hình mạng nơ-ron học sâu kết hợp với các ràng buộc vật lý để tìm mặt cắt đập tối ưu
- **Giao diện người dùng**: Thiết kế tối giản, sạch sẽ theo phong cách Apple
- **Trực quan hóa**: Hiển thị sơ đồ lực tác dụng và biểu đồ hàm mất mát tương tác
- **Báo cáo**: Xuất báo cáo dạng Excel
- **Cơ sở dữ liệu**: Lưu trữ và quản lý kết quả tính toán
- **Triển khai miễn phí**: Tương thích với Streamlit Cloud

## Truy cập ứng dụng

Ứng dụng có thể được triển khai lên Streamlit Cloud bằng cách làm theo hướng dẫn trong phần "Triển khai" bên dưới.

## Cài đặt cục bộ

### Yêu cầu

- Python 3.9 trở lên
- PyTorch 1.13.1
- Các thư viện được liệt kê trong `requirements.txt`

### Cài đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/dam-optimizer-pinns.git
cd dam-optimizer-pinns
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Triển khai

### Triển khai lên Streamlit Cloud

1. Đăng nhập vào GitHub và tạo repository mới
2. Tải mã nguồn lên repository
3. Đăng nhập vào [Streamlit Cloud](https://streamlit.io/cloud)
4. Nhấp vào "New app"
5. Chọn repository GitHub của bạn
6. Nhấp vào "Deploy"

## Cấu trúc dự án

```
dam_optimizer_pinns/
├── app.py                  # Ứng dụng Streamlit chính
├── database.py             # Mô-đun cơ sở dữ liệu
└── requirements.txt        # Các thư viện cần thiết
```

## Sử dụng

1. Nhập các thông số đầu vào:
   - Chiều cao đập `H (m)`
   - Trọng lượng riêng bê tông `γ_bt (T/m³)`
   - Trọng lượng riêng nước `γ_n (T/m³)`
   - Hệ số ma sát `f`
   - Cường độ kháng cắt `C (T/m²)`
   - Hệ số ổn định yêu cầu `Kc`
   - Hệ số áp lực thấm `α1`

2. Nhấn nút "Tính toán tối ưu" để thực hiện tính toán

3. Xem kết quả:
   - Tham số tối ưu (`n`, `m`, `xi`)
   - Diện tích mặt cắt tối ưu (`A`)
   - Hệ số ổn định (`K`)
   - Ứng suất mép thượng lưu (`σ`)
   - Sơ đồ lực tác dụng
   - Biểu đồ hàm mất mát

4. Xuất báo cáo dạng Excel

5. Lưu kết quả vào cơ sở dữ liệu và xem lịch sử tính toán

## Lý thuyết

### Physics-Informed Neural Networks (PINNs)

PINNs là một phương pháp kết hợp giữa mạng nơ-ron học sâu và các ràng buộc vật lý. Trong ứng dụng này, PINNs được sử dụng để tìm các tham số tối ưu của mặt cắt đập bê tông trọng lực, đảm bảo các điều kiện ổn định và an toàn, đồng thời tối thiểu hóa diện tích mặt cắt.

### Các ràng buộc vật lý

1. **Điều kiện ổn định trượt**: Hệ số ổn định K ≥ Kc
2. **Điều kiện không kéo**: Ứng suất mép thượng lưu σ ≤ 0
3. **Tối thiểu hóa diện tích mặt cắt**: Giảm thiểu lượng bê tông sử dụng

## Giấy phép

MIT License
