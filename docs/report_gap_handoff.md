# Bàn giao các phần còn thiếu trong report

Ngày rà soát: 2026-05-07  
Phạm vi: chỉ các phần **không thuộc Agentic AI Component**. Phần Agentic AI Component đã được xử lý riêng trong `Report/Chapters/5_Agent.tex`.

Tài liệu này map yêu cầu trong file `NLP4B - project requirements.pdf` với report LaTeX hiện tại, rồi liệt kê các phần còn thiếu hoặc viết chưa ổn để các thành viên còn lại tiếp tục hoàn thiện. 

## Tổng quan hiện trạng report

| Phần trong report | File hiện tại | Trạng thái | Vấn đề chính |
|---|---|---|---|
| Introduction / Business Problem | `Report/Chapters/1_Introduction.tex` | Đã có một phần | Motivation ổn, nhưng thiếu success metrics có thể đo lường. |
| Data Management | `Report/Chapters/2_Data.tex` | Đã có một phần | Có mô tả dataset và storage, nhưng cần làm rõ split, licensing, preprocessing và data quality. |
| Model Selection and Optimization | `Report/Chapters/3_ModelSelection_Optimization.tex` | Đã có một phần | Có liệt kê model, nhưng thiếu baseline comparison, tuning và kết quả đánh giá cụ thể. |
| Deployment | `Report/Chapters/4_Deployment.tex` | Thiếu | Chapter hiện đang trống. |
| Agentic AI Component | `Report/Chapters/5_Agent.tex` | Được xử lý riêng | Không nằm trong phạm vi bàn giao này. |
| Continual Learning and Monitoring | `Report/Chapters/6_ContinualLearning.tex` | Tương đối đầy đủ | Concept tốt, nhưng nên nối rõ hơn với log/field thật của backend. |
| Privacy, Robustness, Ethics | Chưa thấy chapter/section riêng | Thiếu | Đây là yêu cầu bắt buộc trong rubric. |
| Project Management and Teamwork | Chưa thấy chapter/section riêng | Thiếu | Đây là yêu cầu bắt buộc trong rubric. |
| Development Infrastructure and Tooling | Đang nằm rải rác trong README/docs | Thiếu trong report | Rubric yêu cầu report mô tả professional software development practices. |

## Ma trận đối chiếu rubric

Priority:

- `P0`: cần bổ sung, có rủi ro mất điểm rõ nếu thiếu.
- `P1`: nên bổ sung để report đầy đủ và thuyết phục hơn.
- `P2`: đã tương đối ổn, chỉ cần tinh chỉnh nếu còn thời gian.

| Yêu cầu trong PDF | Report hiện tại | Priority | Cần viết thêm |
|---|---|---:|---|
| Business context and motivation | Đã có trong Introduction | P2 | Giữ lại, nhưng nên làm value proposition đo lường được hơn. |
| Target users / stakeholders | Đã có trong Introduction | P2 | Tạm ổn; có thể thêm một câu về tác động workflow cho từng nhóm user. |
| Why NLP is required | Đã có trong Introduction | P2 | Tạm ổn; nên nối rõ hơn với translation, intent parsing và embedding alignment. |
| Business success metrics | Thiếu | P0 | Thêm metric như thời gian tìm kiếm giảm bao nhiêu, số giờ xem video thủ công tiết kiệm, task completion rate. |
| Technical success metrics | Yếu | P0 | Thêm Recall@K, MRR, nDCG@10, P95 latency, zero-result rate, API availability. |
| Development infrastructure and tooling | Thiếu trong report | P1 | Viết về Python, Git, cấu trúc module, `requirements.txt`, Docker/compose nếu có, tests, `.env`. |
| Data source and licensing | Chưa rõ | P1 | Nêu LongVALE, video từ YouTube, cách dùng public benchmark và các giả định/licensing constraints. |
| Dataset size and languages | Có một phần | P1 | Giữ thông tin 1,200 videos / 13,400 pairs; nói rõ hỗ trợ query tiếng Việt và tiếng Anh. |
| Preprocessing and cleaning | Có nhưng còn khái quát | P1 | Mô tả rõ các bước: download, ffprobe, keyframe extraction, embedding, object detection, OCR, Azure upload, Qdrant upsert. |
| Train/validation/test split | Yếu | P0 | Giải thích vì dùng frozen/zero-shot models nên không có training split truyền thống; cần định nghĩa evaluation subset/held-out query set. |
| Missing/noisy/biased data handling | Có một phần | P1 | Bổ sung failed OCR/detection logs, missing artifact handling, YouTube/source bias. |
| Model architecture description | Có một phần | P1 | Phần model list ổn; cần nói rõ các model liên kết qua late fusion như thế nào. |
| Training procedure | Yếu / không áp dụng trực tiếp | P1 | Nêu hệ thống dùng pretrained models, không train end-to-end; thay vào đó là indexing và evaluation pipeline. |
| Hyperparameter tuning | Thiếu | P0 | Viết về tuning keyframe threshold, retrieval `top_k`, RRF constant, routing weights, rerank coefficients. |
| Baseline comparison | Thiếu | P0 | So sánh ít nhất SigLIP-only, heuristic-only, agentic-only, fused retrieval nếu có kết quả. |
| Evaluation results | Yếu | P0 | Lấy số liệu từ `NLP4B/data-processing/output/evaluation/*.csv` và `metrics_report.*`. |
| Error analysis | Có nhưng còn khái quát | P1 | Ví dụ failure case hiện tại ổn; nên thêm lỗi từ kết quả evaluation thật nếu có. |
| Accuracy vs speed tradeoff | Yếu | P1 | Thêm latency comparison giữa agentic, heuristic và fused strategies. |
| Complexity vs maintainability | Yếu | P2 | Giải thích vì sao tách thành offline pipeline, embedding service, backend và UI. |
| Working inference pipeline | Có rải rác | P1 | Mô tả FastAPI `/search`, Streamlit UI, Azure embedding service và Qdrant dependency. |
| Input/output formats | Thiếu trong Deployment | P0 | Thêm request/response ví dụ cho `/search` và các field UI dùng để render result. |
| Deployment challenges | Thiếu | P0 | Viết về secrets, model hosting, CPU-only Azure VM, Qdrant dependency, latency, model version compatibility. |
| Agent architecture | Được xử lý riêng | N/A | Đã thuộc `5_Agent.tex`. |
| Continual learning strategy | Đã có | P2 | Tốt; nếu có thời gian, nối với field thật trong response/log. |
| Monitoring metrics | Đã có | P2 | Tốt; nên align threshold với Deployment chapter sau khi viết xong. |
| Drift risks and mitigation | Đã có | P2 | Tạm ổn. |
| Privacy and PII handling | Thiếu | P0 | Viết về faces, YouTube metadata, OCR text, user query logs, IP/session data nếu có, retention policy. |
| Robustness to noisy/adversarial input | Thiếu | P0 | Viết về typo, query đa ngôn ngữ, OCR noise, prompt-injection-like query, out-of-domain input. |
| Project plan / timeline | Thiếu | P1 | Thêm bảng timeline theo phase: ingestion, indexing, backend, UI, evaluation, report. |
| Task breakdown / teamwork | Thiếu | P1 | Thêm role split: data, backend, retrieval, UI, evaluation, documentation. |
| Ethics impact statement | Thiếu | P0 | Viết ai được lợi, ai có thể bị ảnh hưởng xấu, surveillance misuse, privacy risks. |
| Bias and fairness risks | Có một phần trong Data | P1 | Mở rộng từ sourcing bias sang visual model bias và Vietnamese translation bias. |
| Explainability | Yếu | P1 | Giải thích evidence tags, source contribution, latency trace và vì sao result dễ hiểu với non-technical stakeholders. |
| Potential misuse | Thiếu | P0 | Nêu misuse như tìm kiếm giám sát, stalking, trích xuất text nhạy cảm trong video. |

## Hướng dẫn chi tiết theo section

### 1. Introduction

Điểm mạnh hiện tại:

- Business motivation rõ: tìm đúng khoảnh khắc trong video dài.
- Đã xác định target users: media archives, security/CCTV, general video consumers.
- Đã giải thích vai trò NLP qua language normalization, intent understanding và embedding alignment.

Thiếu hoặc chưa ổn:

- Chưa có bảng success metrics rõ ràng.
- Business value vẫn định tính, chưa có tiêu chí đo lường.

Gợi ý bổ sung:

- Thêm bảng gồm business metrics và technical metrics.
- Business metrics: average search time reduction, task completion rate, user satisfaction, manual review hours saved.
- Technical metrics: Recall@1/5/10, MRR, nDCG@10, P95 latency, zero-result rate.

### 2. Data Management

Điểm mạnh hiện tại:

- Đã nêu LongVALE dataset và lý do chọn.
- Data storage strategy đã giải thích Qdrant và Azure.
- Known limitations đã nhắc temporal granularity và translation loss.

Thiếu hoặc chưa ổn:

- Chưa nói rõ licensing/source constraints.
- Split justification đang nói vì không có training phase, nhưng vẫn cần mô tả evaluation split.
- Preprocessing còn cần cụ thể hơn và gắn với script thật.
- Nên nói thêm cách xử lý OCR/object detection failed artifacts.

Gợi ý bổ sung:

- Nêu rõ các model chủ yếu là zero-shot/frozen, nên split dùng cho evaluation thay vì supervised training.
- Định nghĩa held-out query/event subset để evaluate.
- Nhắc failure logs và progress CSVs như data-quality controls.
- Thêm bảng pipeline ngắn: input, script, output artifact, downstream consumer.

### 3. Model Selection and Optimization

Điểm mạnh hiện tại:

- Đã nêu các model chính và modality tương ứng.
- Failure cases hợp lý.

Thiếu hoặc chưa ổn:

- Chưa có baseline comparison.
- Chưa có hyperparameter tuning discussion.
- Chưa có bảng evaluation result cụ thể.
- Caption của architecture figure đang trống.
- Cần giải thích late fusion rõ hơn.

Gợi ý bổ sung:

- Thêm baselines: SigLIP-only visual search, caption-only semantic search, heuristic retrieval, agentic retrieval, fused retrieval.
- Dùng evaluation artifacts trong `NLP4B/data-processing/output/evaluation/`.
- Thêm bảng Recall@K, MRR/MAP/nDCG nếu metric có trong `metrics_report.*`.
- Discuss tuning: keyframe redundancy threshold, `top_k`, RRF `k`, routing weights, rerank coefficients.
- Thêm tradeoff paragraph: agentic branch intent-align tốt hơn nhưng tốn LLM/embedding latency; heuristic nhanh hơn và làm fallback tốt.

### 4. Deployment

Hiện trạng:

- Chapter đang trống.

Nội dung bắt buộc nên có:

- User interaction: Streamlit UI gửi natural-language query và render keyframe cards.
- Input/output formats: `POST /search` với `raw_query`, `top_k`, `strategy`; response gồm ranked results, URLs, evidence, latency.
- Runtime boundaries: Streamlit frontend, FastAPI backend, Azure embedding service, Qdrant Cloud, Azure Blob Storage.
- Environment variables: `QDRANT_URL`, `QDRANT_API_KEY`, `EMBEDDING_API_BASE_URL`, LLM provider keys.
- Deployment challenges: secrets, model version compatibility, cold start, CPU-only embedding service, external service failures, latency.
- Scalability: over-fetching, batch embedding, concurrent Qdrant search, cloud vector DB.
- Model versioning: pin model IDs và cần re-index khi embedding model thay đổi.

Gợi ý cấu trúc:

1. Deployment Architecture
2. User Interaction and API Contract
3. Runtime Configuration
4. Latency and Scalability Considerations
5. Deployment Challenges and Limitations

### 6. Continual Learning and Monitoring

Điểm mạnh hiện tại:

- Đây là chapter tương đối tốt nhất trong các phần không thuộc agentic.
- Đã có implicit/explicit feedback, retraining tiers, online metrics và drift risks.

Thiếu hoặc chưa ổn:

- Một số metric còn mang tính concept, chưa nối với field thật của backend.
- Retraining section nên phân biệt rõ hơn giữa fine-tuning model và re-indexing corpus.

Gợi ý bổ sung:

- Nối query logs với các field hiện có trong response: `latency_ms`, `branch`, `evidence`, `total_results`.
- Thêm một câu rằng đa số update thực tế sẽ là corpus/index update chứ không phải full model retraining.

### Privacy, Robustness, and Ethics

Hiện trạng:

- Chưa có phần riêng đủ rõ.

Nội dung cần có:

- PII trong video frames: faces, license plates, names, phone numbers, signage, OCR text.
- PII trong user queries/logs: sensitive search terms, identifiers, IP/session data nếu có log.
- Data minimization: tránh lưu raw user identifiers không cần thiết; aggregate monitoring metrics.
- Security: API keys để trong `.env`, không commit secrets, giới hạn access tới internal embedding service.
- Robustness: typos, Vietnamese/English mixed queries, OCR noise, out-of-domain requests, adversarial prompt-like query text.
- Misuse: surveillance, stalking, trích xuất thông tin riêng tư từ video, tìm kiếm địa điểm nhạy cảm.
- Fairness: bias từ YouTube content, visual model bias, translation bias với tiếng Việt/slang/cultural terms.
- Explainability: evidence sources và latency trace giúp non-technical users hiểu kết quả.

Gợi ý vị trí:

- Tạo chapter mới trước Continual Learning hoặc thêm hai section sau Deployment.
- Không nên chỉ đưa phần này vào appendix vì đây là requirement chính.

### Project Management and Teamwork

Hiện trạng:

- Thiếu.

Nội dung cần có:

- Project timeline.
- Task breakdown và roles.
- Reflection về cách project scale nếu làm như team thật.

Gợi ý bổ sung:

- Một bảng phase: dataset preparation, offline processing, vector indexing, backend retrieval, UI, evaluation, report.
- Một bảng role: data pipeline, embedding/indexing, backend, frontend, evaluation, documentation.
- Một đoạn ngắn về coordination risks: schema changes, model version drift, shared environment variables, integration testing.

## Checklist trước khi nộp

- [ ] Mọi figure/table đều có caption rõ và được nhắc trong text.
- [ ] `Report/Chapters/4_Deployment.tex` không còn trống.
- [ ] Business metrics và technical success metrics được viết rõ.
- [ ] Có baseline comparison và evaluation results.
- [ ] Privacy, robustness, ethics và misuse được đưa vào nội dung chính.
- [ ] Có phần project management/teamwork.
- [ ] Không có secrets, hard-coded credentials trong screenshot, code snippet hoặc nội dung report.
- [ ] PDF compile không có fatal LaTeX errors.
- [ ] Slides khớp với nội dung cuối cùng của written report.
