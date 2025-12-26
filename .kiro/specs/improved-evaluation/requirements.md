# Requirements Document

## Introduction

Cải thiện cơ chế đánh giá (evaluation) trong file `train_qplex_mate.py` để giải quyết hai vấn đề chính: (1) chuyển từ 1 nhóm agent sang 2 nhóm agent riêng biệt, và (2) cải thiện độ ổn định và hội tụ của kết quả đánh giá hiện đang có độ lệch cao giữa các lần chạy 2000 episodes.

## Glossary

- **Evaluation System**: Hệ thống đánh giá hiệu suất của agent trong môi trường MATE
- **Agent Group**: Nhóm các camera agents được quản lý và đánh giá cùng nhau
- **Episode**: Một lần chạy hoàn chỉnh từ reset đến terminated/truncated
- **Convergence**: Độ hội tụ của kết quả đánh giá, thể hiện qua độ lệch chuẩn thấp giữa các lần đánh giá
- **QPLEX Learner**: Thuật toán học tăng cường multi-agent được sử dụng
- **MATE Environment**: Môi trường Multi-Agent Tracking Environment

## Requirements

### Requirement 1

**User Story:** Là một nhà nghiên cứu RL, tôi muốn đánh giá 2 nhóm agent riêng biệt thay vì 1 nhóm, để có thể so sánh hiệu suất giữa các nhóm và phân tích chi tiết hơn.

#### Acceptance Criteria

1. WHEN đánh giá được thực hiện, THE Evaluation System SHALL tạo và quản lý 2 Agent Groups riêng biệt
2. THE Evaluation System SHALL thu thập metrics riêng biệt cho mỗi Agent Group
3. THE Evaluation System SHALL tính toán và lưu trữ kết quả đánh giá cho từng Agent Group một cách độc lập
4. THE Evaluation System SHALL cung cấp khả năng so sánh metrics giữa 2 Agent Groups
5. WHEN lưu kết quả đánh giá, THE Evaluation System SHALL ghi rõ metrics của từng Agent Group vào file JSON

### Requirement 2

**User Story:** Là một nhà nghiên cứu RL, tôi muốn kết quả đánh giá có độ hội tụ cao và ổn định hơn, để có thể tin tưởng vào kết quả và đưa ra quyết định chính xác về hiệu suất của model.

#### Acceptance Criteria

1. THE Evaluation System SHALL thực hiện warm-up episodes trước khi bắt đầu thu thập metrics chính thức
2. THE Evaluation System SHALL sử dụng multiple evaluation runs với seeds khác nhau để giảm variance
3. THE Evaluation System SHALL tính toán confidence intervals cho các metrics chính
4. THE Evaluation System SHALL loại bỏ outliers sử dụng phương pháp thống kê (IQR method)
5. WHEN số lượng episodes lớn, THE Evaluation System SHALL chia thành các batches nhỏ hơn và tính trung bình có trọng số
6. THE Evaluation System SHALL theo dõi và báo cáo coefficient of variation (CV) để đánh giá độ ổn định
7. THE Evaluation System SHALL lưu trữ raw data của tất cả episodes để phân tích sau này

### Requirement 3

**User Story:** Là một nhà nghiên cứu RL, tôi muốn có thể cấu hình linh hoạt các tham số đánh giá, để có thể điều chỉnh theo nhu cầu thí nghiệm cụ thể.

#### Acceptance Criteria

1. THE Evaluation System SHALL cho phép cấu hình số lượng warm-up episodes qua config file
2. THE Evaluation System SHALL cho phép cấu hình số lượng evaluation runs và seeds qua config file
3. THE Evaluation System SHALL cho phép cấu hình batch size cho evaluation qua config file
4. THE Evaluation System SHALL cho phép bật/tắt outlier removal qua config file
5. THE Evaluation System SHALL cho phép cấu hình confidence level cho confidence intervals qua config file

### Requirement 4

**User Story:** Là một nhà nghiên cứu RL, tôi muốn có logging và visualization tốt hơn cho kết quả đánh giá, để dễ dàng theo dõi và phân tích tiến trình training.

#### Acceptance Criteria

1. THE Evaluation System SHALL log chi tiết kết quả của từng evaluation run
2. THE Evaluation System SHALL log confidence intervals cùng với mean values
3. THE Evaluation System SHALL log coefficient of variation để đánh giá độ ổn định
4. THE Evaluation System SHALL lưu raw evaluation data vào file CSV để phân tích sau
5. THE Evaluation System SHALL tạo summary statistics bao gồm min, max, median, Q1, Q3 cho mỗi metric
