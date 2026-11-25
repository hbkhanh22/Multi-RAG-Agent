# evals/

Bộ câu hỏi đánh giá. Cách chạy: [src/evaluation/README.md](../src/evaluation/README.md).

| Mục | Nội dung |
| :-- | :-- |
| [datasets/sport.jsonl](datasets/sport.jsonl) | 18 câu thể thao |
| [datasets/tech.jsonl](datasets/tech.jsonl) | 6 câu công nghệ |
| `results/` | Kết quả mỗi lần chạy, không đưa lên git |

Mỗi dòng là một câu (số trang tính từ 0):

```json
{"id": "sport-ball", "question": "...", "domain": "sport", "reference": "đáp án mẫu",
 "evidence": [{"source": "Laws of the Game", "pages": [27]}]}
```

## Kết quả ngày 25/11/2026

24 câu, image Docker hiện tại (pgvector, ONNX int8), LLM thật. Judge cùng model với LLM trả lời nên điểm có thể lạc quan.

| Nhóm | Kết quả |
| :-- | :-- |
| Định tuyến đúng | 24/24 |
| Truy xuất | Hit rate 0.96 (23/24), MRR 0.88, precision 0.55 |
| Judge (1–5) | Faithfulness 4.96, relevance 5.0, completeness 4.88, correctness 5.0 |
| Viết lại | 2 câu được viết lại, 1 câu đạt sau đó |
| Thời gian mỗi câu | Trung bình 13.5 giây, lâu nhất 43.2 giây |

- Câu trượt truy xuất: `sport-kickoff-goal`.
- Thời gian gồm phần chờ model kiểm duyệt đang quá hạn (tới 5 giây mỗi câu).
- Bộ câu hỏi không gồm web, tệp đính kèm, MCP và câu hỏi nhiều miền. Các phần này mới thử vài câu.
