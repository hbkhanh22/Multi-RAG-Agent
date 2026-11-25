# frontend/

Giao diện chat React + Vite.

| Mục | Chức năng |
| :-- | :-- |
| [src/](src/README.md) | Mã nguồn |
| [index.html](index.html) | Trang gốc |
| [package.json](package.json) | Thư viện và lệnh npm |
| [vite.config.js](vite.config.js) | Cấu hình build |
| [Dockerfile](Dockerfile) | Image web: build tĩnh, phục vụ bằng nginx |
| [nginx.conf](nginx.conf) | Phục vụ trang, chuyển `/api` tới API (không đệm SSE), giới hạn tải lên 50 MB |
| [.env.example](.env.example), [.env.production](.env.production) | `VITE_API_BASE_URL`: địa chỉ API khi dev; `.` (cùng origin) khi build |

```bash
npm install
npm run dev       # http://localhost:5173, API ở http://localhost:8000
npm run build     # tạo dist/
```

## Build cho Docker

| `WEB_DIST` trong `.env` | Cách làm |
| :-- | :-- |
| `build` (mặc định) | Image tự build, cần tải được `node:22-alpine` |
| `prebuilt` | Image chép `dist/` có sẵn. Chạy `npm run build` **xong** rồi mới `docker compose up -d --build web` |

Sửa giao diện mà không thấy thay đổi: thường do chưa build lại `dist/` trước khi build image.
