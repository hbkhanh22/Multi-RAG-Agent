# scripts/

| File | Chức năng | Lệnh |
| :-- | :-- | :-- |
| [export_onnx.py](export_onnx.py) | Xuất embedding và reranker sang ONNX int8 (khoảng 0.57 GB mỗi mô hình thay vì 2.2 GB). Cần `requirements.txt`. Mô hình đã xuất đang được dùng; script chưa chạy lại ngày 25/11 | `python scripts/export_onnx.py` |
| [sync_models.py](sync_models.py) | Đẩy hoặc kéo mô hình ONNX với S3. **Chưa kiểm** | `python scripts/sync_models.py push`, hoặc `pull --if-missing` |
| [eval_gate.py](eval_gate.py) | Trả lỗi khi chất lượng truy xuất dưới ngưỡng. Job CI tương ứng cần runner riêng, mặc định tắt | `python scripts/eval_gate.py <thư mục kết quả> --min-hit 0.9 --min-mrr 0.8` |

`export_onnx.py --only reranker` xuất một mô hình; `--verify` so kết quả int8 với bản gốc.
