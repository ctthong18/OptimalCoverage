# Requirements Document

## Introduction

This document specifies the requirements for integrating the QPLEX HRL (Hierarchical Reinforcement Learning) evaluation approach from the MATE examples into the existing new_qplex implementation. The integration aims to enhance the new_qplex algorithm with hierarchical decision-making capabilities and comprehensive evaluation metrics from the MATE framework.

## Glossary

- **QPLEX**: Q-value decomposition algorithm for multi-agent reinforcement learning
- **HRL**: Hierarchical Reinforcement Learning - a learning approach with multiple levels of decision-making
- **MATE**: Multi-Agent Tracking Environment - the simulation environment for camera-target tracking
- **new_qplex**: The optimized QPLEX implementation in the current codebase
- **HierarchicalCamera**: Wrapper that enables hierarchical target selection for camera agents
- **EvaluationSystem**: The comprehensive evaluation infrastructure in evaluation_utils.py
- **CameraAgent**: Agent that controls camera movements and target selection
- **TargetAgent**: Agent that controls target movements (typically greedy or heuristic)

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to integrate HRL capabilities into new_qplex, so that I can leverage hierarchical decision-making for improved multi-agent coordination.

#### Acceptance Criteria

1. WHEN the system initializes THEN the new_qplex SHALL support hierarchical camera wrapper integration
2. WHEN agents make decisions THEN the system SHALL use multi-level target selection with frame skipping
3. WHEN processing observations THEN the system SHALL handle hierarchical observation spaces correctly
4. WHEN training occurs THEN the system SHALL maintain compatibility with existing new_qplex network architecture
5. WHEN evaluating performance THEN the system SHALL track hierarchical decision metrics

### Requirement 2

**User Story:** As a developer, I want to adapt the MATE HRL evaluation approach, so that I can comprehensively assess new_qplex performance with hierarchical metrics.

#### Acceptance Criteria

1. WHEN evaluation runs THEN the system SHALL collect coverage rates, transport rates, and selection metrics
2. WHEN calculating statistics THEN the system SHALL use the existing StatisticsCalculator for outlier removal and confidence intervals
3. WHEN comparing groups THEN the system SHALL perform statistical significance testing with effect size calculation
4. WHEN logging results THEN the system SHALL save both JSON summaries and CSV raw data
5. WHEN displaying results THEN the system SHALL show hierarchical-specific metrics alongside standard reward metrics

### Requirement 3

**User Story:** As a researcher, I want to configure HRL parameters, so that I can experiment with different hierarchical settings for optimal performance.

#### Acceptance Criteria

1. WHEN configuring the system THEN the user SHALL specify multi_selection mode (boolean)
2. WHEN setting frame parameters THEN the user SHALL configure frame_skip values for temporal hierarchy
3. WHEN defining rewards THEN the user SHALL set reward coefficients for coverage and transport rates
4. WHEN choosing evaluation THEN the user SHALL configure enhanced observation modes
5. WHEN selecting opponents THEN the user SHALL specify target agent factory functions

### Requirement 4

**User Story:** As a system integrator, I want to create HRL-compatible agents, so that new_qplex can work with the MATE hierarchical environment.

#### Acceptance Criteria

1. WHEN creating agents THEN the system SHALL implement HRLQPLEXCameraAgent interface
2. WHEN processing actions THEN the system SHALL convert discrete selections to continuous camera actions
3. WHEN managing state THEN the system SHALL handle hidden states and action mapping correctly
4. WHEN resetting episodes THEN the system SHALL properly initialize hierarchical components
5. WHEN executing actions THEN the system SHALL use the HierarchicalCamera executor for primitive actions

### Requirement 5

**User Story:** As an evaluator, I want to run comprehensive HRL evaluations, so that I can compare new_qplex performance against baseline algorithms.

#### Acceptance Criteria

1. WHEN starting evaluation THEN the system SHALL run warm-up episodes to stabilize hidden states
2. WHEN conducting runs THEN the system SHALL execute multiple evaluation runs with different seeds
3. WHEN aggregating results THEN the system SHALL calculate hierarchical metrics with statistical confidence
4. WHEN comparing algorithms THEN the system SHALL perform group comparisons with significance testing
5. WHEN completing evaluation THEN the system SHALL generate comprehensive reports with HRL-specific insights

### Requirement 6

**User Story:** As a configuration manager, I want to set up HRL environments, so that new_qplex can train and evaluate in hierarchical multi-agent scenarios.

#### Acceptance Criteria

1. WHEN creating environments THEN the system SHALL wrap base MATE environments with hierarchical components
2. WHEN applying transformations THEN the system SHALL use RelativeCoordinates and RescaledObservation wrappers
3. WHEN configuring rewards THEN the system SHALL apply AuxiliaryCameraRewards with specified coefficients
4. WHEN setting up multi-agent THEN the system SHALL use RLlibMultiAgentAPI and centralized training wrappers
5. WHEN registering environments THEN the system SHALL make environments available for Ray/RLlib integration

### Requirement 7

**User Story:** As a trainer, I want to train new_qplex with HRL capabilities, so that I can develop hierarchical multi-agent policies.

#### Acceptance Criteria

1. WHEN initializing training THEN the system SHALL configure QPLEX with hierarchical observation and action spaces
2. WHEN setting hyperparameters THEN the system SHALL use appropriate learning rates, buffer sizes, and network architectures
3. WHEN running training THEN the system SHALL handle multi-discrete action spaces with proper exploration
4. WHEN checkpointing THEN the system SHALL save hierarchical policy states for evaluation
5. WHEN monitoring progress THEN the system SHALL track hierarchical training metrics and convergence

### Requirement 8

**User Story:** As a performance analyst, I want to analyze HRL evaluation results, so that I can understand the effectiveness of hierarchical decision-making in new_qplex.

#### Acceptance Criteria

1. WHEN analyzing coverage THEN the system SHALL report target coverage rates and selection efficiency
2. WHEN examining transport THEN the system SHALL measure cargo transport rates and delivery success
3. WHEN evaluating selection THEN the system SHALL track valid vs invalid target selections
4. WHEN comparing performance THEN the system SHALL show statistical significance of improvements
5. WHEN generating insights THEN the system SHALL provide actionable recommendations for parameter tuning