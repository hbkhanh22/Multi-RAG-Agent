# src/evaluation/

| File | Chức năng |
| :-- | :-- |
| [judge.py](judge.py) | `AnswerJudge` chấm câu trả lời; `AnswerRefiner` viết lại theo nhận xét |
| [metrics.py](metrics.py) | Hit rate, MRR, precision; tổng hợp kết quả |
| [ragas_eval.py](ragas_eval.py) | Độ đo RAGAS, chưa kiểm lại |
| [run.py](run.py) | Lệnh đánh giá offline |

## Online: trong mỗi câu hỏi

Judge chấm 1–5 theo ba tiêu chí: **faithfulness** (bám ngữ cảnh), **relevance** (đúng câu hỏi), **completeness** (đủ ý). Có tiêu chí dưới `SELF_EVAL_MIN_SCORE` thì viết lại và chấm lại, tối đa `SELF_EVAL_MAX_ROUNDS` vòng. Trong lần đo 25/11/2026, 2/24 câu được viết lại và 1 câu đạt sau đó.

Chi phí: thêm 1 lần gọi LLM, hoặc 3 lần nếu phải viết lại. Judge không bắt được mọi lỗi, vì cũng là một LLM.

## Offline: trên bộ câu hỏi mẫu

```bash
python -m src.evaluation.run --dataset evals/datasets/sport.jsonl evals/datasets/tech.jsonl
python -m src.evaluation.run --dataset evals/datasets/sport.jsonl --no-refine      # không viết lại
python -m src.evaluation.run --dataset evals/datasets/sport.jsonl --retrieval-only # chỉ đo truy xuất
python -m src.evaluation.run --only sport-gk-8s sport-subs                         # vài câu
```

Định dạng bộ câu hỏi và kết quả: [evals/README.md](../../evals/README.md).
