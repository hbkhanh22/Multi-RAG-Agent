# src/auth/

Tài khoản, phiên đăng nhập và phân quyền. Bật bằng `AUTH_ENABLED=true` và `JWT_SECRET` trong `.env` (mặc định tắt). Endpoint ở [api/auth.py](../../api/auth.py), giao diện ở [frontend/src](../../frontend/src/README.md#đăng-nhập).

| File | Chức năng |
| :-- | :-- |
| [users.py](users.py) | Tài khoản, vai trò `user` và `admin` |
| [passwords.py](passwords.py) | Băm mật khẩu bằng scrypt |
| [tokens.py](tokens.py) | Access token JWT (HS256, 15 phút) và refresh token xoay vòng (14 ngày) |
| [cli.py](cli.py) | Lệnh quản trị tài khoản trên máy chủ |
| [google.py](google.py) | Google Sign-In. **Chưa cấu hình, chưa chạy với Google thật** (chỉ test giả lập) |

## Cấp quyền admin

Tự đăng ký luôn tạo tài khoản thường và không đăng ký được địa chỉ trong `ADMIN_EMAILS`. Chỉ có hai cách thành admin: lệnh dưới đây, hoặc đăng nhập Google bằng địa chỉ trong `ADMIN_EMAILS`.

```bash
docker compose exec api python -m src.auth.cli admin ban@example.com           # tạo admin, hoặc nâng quyền kèm mật khẩu mới
docker compose exec api python -m src.auth.cli claim ban@example.com           # nhận hội thoại có từ trước khi bật đăng nhập
docker compose exec api python -m src.auth.cli reset-password ban@example.com  # đặt lại mật khẩu, đăng xuất mọi thiết bị
docker compose exec api python -m src.auth.cli demote ban@example.com          # hạ xuống tài khoản thường
docker compose exec api python -m src.auth.cli list
```

Nâng quyền luôn đổi mật khẩu và thu hồi phiên, phòng khi người khác đã đăng ký địa chỉ đó trước. Vì cùng lý do, khi chủ email liên kết Google thì mật khẩu cũ của tài khoản thường bị xóa.

## Kết quả kiểm tra 25/11/2026

Image Docker, `AUTH_ENABLED=true`, PostgreSQL thật, 2 worker API. Backend **61/61**, giao diện trên Chrome **40/40**, test tự động `test_security` **24/24**.

| Nhóm | Đã kiểm |
| :-- | :-- |
| Không có token | 15 endpoint đều trả 401 |
| JWT giả | Sửa nội dung, `alg: none`, sai khóa, HS512, hết hạn, thiếu `exp`, sai `iss`, sai `typ`, chưa hiệu lực: đều 401 |
| Refresh token | Xoay vòng; dùng lại token cũ sau 10 giây thì thu hồi cả phiên; 12 yêu cầu đồng thời; cookie hỏng trả 401 |
| Tách dữ liệu | Đọc, xóa, tiếp tục hội thoại; dùng, xóa tệp của người khác: đều bị từ chối |
| Admin | Không tự đăng ký được email admin; người đăng ký trước không thành admin; hạ quyền có hiệu lực ngay |
| Giao diện | Đăng ký, đăng nhập, lỗi nhập liệu, tải lại, nhiều tab, hết hạn và đăng nhập lại, đăng xuất, đổi tài khoản, mất kết nối, điện thoại, giao diện tối |

Token vừa xoay vòng vẫn dùng được trong 10 giây (`REFRESH_REUSE_GRACE_SECONDS`) để mở nhiều tab hoặc tải lại nhanh không bị đăng xuất.

## Vấn đề chưa sửa

| Mức | Vấn đề |
| :-- | :-- |
| Trung bình | Không xác minh email và không có quên mật khẩu qua email: admin đặt lại bằng lệnh. Ai cũng tự đăng ký được trừ khi đặt `ALLOW_REGISTRATION=false` |
| Trung bình | Giới hạn dò mật khẩu tính riêng từng worker, nên với 2 worker số lần thử tối đa có thể gấp đôi |
| Trung bình | Access token vẫn dùng được tới 15 phút sau khi đăng xuất (trừ quyền admin, được kiểm lại mỗi lần gọi) |
| Thấp | `/docs`, `/api/stats`, `/api/mcp/status` không cần token; `/metrics` cũng vậy nhưng nginx chặn từ ngoài |
