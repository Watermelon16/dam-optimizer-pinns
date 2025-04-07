# Hướng dẫn triển khai ứng dụng tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs

## Giới thiệu

Tài liệu này hướng dẫn cách triển khai ứng dụng tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng mô hình Physics-Informed Neural Networks (PINNs) lên Streamlit Cloud. Ứng dụng này đã được thiết kế để tận dụng sức mạnh của học sâu kết hợp với các ràng buộc vật lý để tìm ra mặt cắt tối ưu nhất.

## Yêu cầu

- Tài khoản GitHub (có thể đăng nhập bằng passkey)
- Mã nguồn ứng dụng (đã được cung cấp)

## Các bước triển khai

### 1. Tạo repository trên GitHub

1. Đăng nhập vào GitHub bằng passkey của bạn
2. Nhấp vào nút "+" ở góc trên bên phải và chọn "New repository"
3. Điền thông tin:
   - **Repository name**: `dam-optimizer-pinns`
   - **Description** (tùy chọn): `Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs`
   - **Visibility**: Chọn "Public"
   - **Initialize this repository with**: Không chọn gì
4. Nhấp vào "Create repository"

### 2. Tải mã nguồn lên GitHub

1. Giải nén file chứa mã nguồn mà tôi đã gửi cho bạn
2. Mở terminal/command prompt và di chuyển đến thư mục đã giải nén:
   ```bash
   cd đường_dẫn_đến_thư_mục/dam_optimizer_pinns
   ```
3. Khởi tạo Git repository và đẩy mã nguồn lên GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/username_của_bạn/dam-optimizer-pinns.git
   git push -u origin main
   ```

   Thay `username_của_bạn` bằng tên người dùng GitHub của bạn.

### 3. Đăng ký tài khoản Streamlit Cloud

1. Truy cập [Streamlit Cloud](https://streamlit.io/cloud)
2. Nhấp vào "Sign up" nếu bạn chưa có tài khoản
3. Chọn "Continue with GitHub" để đăng nhập bằng tài khoản GitHub của bạn
4. Làm theo các bước để hoàn tất đăng ký

### 4. Triển khai ứng dụng lên Streamlit Cloud

1. Sau khi đăng nhập vào Streamlit Cloud, nhấp vào "New app"
2. Trong phần "Repository", chọn repository `dam-optimizer-pinns` của bạn
3. Trong phần "Branch", chọn `main`
4. Trong phần "Main file path", nhập `app.py`
5. Nhấp vào "Deploy"

### 5. Kiểm tra ứng dụng

1. Streamlit Cloud sẽ tự động xây dựng và triển khai ứng dụng của bạn
2. Quá trình này có thể mất vài phút
3. Khi hoàn tất, bạn sẽ thấy ứng dụng của mình chạy trên URL dạng `https://username-dam-optimizer-pinns.streamlit.app`

## Xử lý sự cố

### Nếu gặp lỗi khi triển khai

1. Kiểm tra logs lỗi bằng cách nhấp vào "Manage app" và xem phần "Logs"
2. Đảm bảo rằng bạn đã đẩy tất cả các file lên GitHub, bao gồm `app.py`, `database.py` và `requirements.txt`
3. Nếu gặp lỗi liên quan đến PyTorch, hãy kiểm tra file `requirements.txt` để đảm bảo nó bao gồm dòng:
   ```
   -f https://download.pytorch.org/whl/cpu/torch_stable.html
   torch==1.13.1
   ```
4. Nếu vẫn gặp vấn đề, hãy thử các bước sau:
   - Kiểm tra xem repository của bạn có công khai (public) không
   - Đảm bảo rằng bạn đã cấp quyền cho Streamlit Cloud truy cập vào repository của bạn
   - Thử tạo lại ứng dụng

## Bảo trì và cập nhật

### Cập nhật mã nguồn

1. Khi bạn muốn cập nhật ứng dụng, chỉ cần thay đổi mã nguồn trên máy tính của bạn
2. Commit và push các thay đổi lên GitHub:
   ```bash
   git add .
   git commit -m "Cập nhật mã nguồn"
   git push
   ```
3. Streamlit Cloud sẽ tự động phát hiện các thay đổi và cập nhật ứng dụng

## Kết luận

Bạn đã hoàn thành việc triển khai ứng dụng tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs lên Streamlit Cloud. Ứng dụng của bạn giờ đây có thể truy cập từ bất kỳ đâu và bất kỳ thiết bị nào có kết nối internet.

Nếu bạn muốn thêm tính năng hoặc cải thiện ứng dụng, bạn có thể cập nhật mã nguồn và đẩy lên GitHub. Streamlit Cloud sẽ tự động cập nhật ứng dụng của bạn.
