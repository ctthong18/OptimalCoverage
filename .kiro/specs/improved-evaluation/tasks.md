# Implementation Plan

- [x] 1. Tạo module evaluation utilities với StatisticsCalculator





  - Implement class `StatisticsCalculator` với các static methods
  - Implement `remove_outliers()` method sử dụng IQR và Z-score methods
  - Implement `calculate_confidence_interval()` method sử dụng t-distribution
  - Implement `calculate_convergence_metrics()` method (CV, stability score)
  - Implement `calculate_summary_statistics()` method (min, max, median, Q1, Q3)
  - _Requirements: 2.4, 2.6_

- [x] 1.1 Viết unit tests cho StatisticsCalculator


  - Test outlier removal với known data và edge cases
  - Test confidence interval calculation
  - Test convergence metrics calculation
  - _Requirements: 2.4, 2.6_

- [x] 2. Tạo data models cho evaluation results





  - Implement `EvaluationConfig` dataclass với validation
  - Implement `GroupResults` dataclass
  - Implement `AggregatedResults` dataclass
  - Implement `ComparisonResults` dataclass
  - Add serialization methods (to_dict, from_dict)
  - _Requirements: 1.2, 1.3, 2.7_

- [x] 3. Implement GroupEvaluator class





  - Tạo class `GroupEvaluator` với constructor nhận group_id
  - Implement `_run_single_episode()` method để chạy một episode và thu thập metrics
  - Implement `evaluate_group()` method để chạy n_episodes với seed cụ thể
  - Add proper error handling và retry logic
  - Add metrics collection: episode_rewards, episode_lengths, coverage_rates, transport_rates
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 3.1 Viết unit tests cho GroupEvaluator


  - Test single episode execution với mock environment
  - Test metrics collection
  - Test error handling và retry logic
  - _Requirements: 1.1, 1.2_
-

- [x] 4. Implement EvaluationLogger class




  - Tạo class `EvaluationLogger` với log_dir parameter
  - Implement `save_json()` method để lưu results dưới dạng JSON
  - Implement `save_csv()` method để lưu raw data dưới dạng CSV
  - Implement `print_summary()` method để in summary ra console với formatting đẹp
  - Implement `log_results()` method tổng hợp tất cả logging operations
  - Add proper directory creation và error handling
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
-

- [x] 4.1 Viết unit tests cho EvaluationLogger






  - Test JSON serialization và file writing
  - Test CSV writing với different data formats
  - Test directory creation
  - _Requirements: 4.4, 4.5_

- [x] 5. Implement ImprovedEvaluator class




  - Tạo class `ImprovedEvaluator` với EvaluationConfig
  - Implement `run_warmup_episodes()` method để chạy warm-up episodes
  - Implement `_evaluate_single_run()` method để chạy một evaluation run cho cả 2 groups
  - Implement `_aggregate_results()` method để aggregate results từ multiple runs
  - Implement `_compare_groups()` method để so sánh 2 groups (t-test, effect size)
  - Implement main `evaluate()` method orchestrating toàn bộ flow
  - Add progress tracking và logging
  - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.5_

- [x] 5.1 Viết integration tests cho ImprovedEvaluator







  - Test full evaluation flow với mock learners và environment
  - Test warm-up episodes
  - Test multiple runs với different seeds
  - Test aggregation và comparison
  - _Requirements: 1.1, 2.1, 2.2, 2.3_

- [x] 6. Update train_qplex_mate.py để sử dụng ImprovedEvaluator



  - Import ImprovedEvaluator và related classes
  - Tạo 2 QPLEXLearner instances cho 2 groups (có thể clone từ main learner)
  - Replace hàm `evaluate_agent()` cũ bằng `ImprovedEvaluator.evaluate()`
  - Update evaluation call trong training loop
  - Update logging để hiển thị results của cả 2 groups
  - Ensure backward compatibility với old checkpoints
  - _Requirements: 1.1, 1.5, 4.1, 4.2_

- [x] 7. Update config files với evaluation settings mới







  - Update `configs/qplex_4v4_9.yaml` với evaluation section mới
  - Update `configs/qplex_4v8_0.yaml` với evaluation section mới
  - Add default values cho tất cả evaluation parameters
  - Add comments giải thích các parameters
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
- [x] 8. Tạo evaluation utilities script







- [ ] 8. Tạo evaluation utilities script

  - Tạo script `evaluate_checkpoint.py` để evaluate saved checkpoints
  - Support loading 2 checkpoints và compare
  - Support custom evaluation config
  - Add visualization options (plots, charts)
  - _Requirements: 1.4, 4.5_

- [ ] 9. Add visualization và plotting

  - Implement plotting functions cho evaluation results
  - Create comparison plots giữa 2 groups
  - Create convergence plots showing stability over runs
  - Create distribution plots với outliers highlighted
  - Save plots vào log directory
  - _Requirements: 4.5_

- [ ] 10. Update documentation

  - Update README.md với evaluation instructions
  - Add docstrings cho tất cả classes và methods
  - Create example usage notebook
  - Document configuration options
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
