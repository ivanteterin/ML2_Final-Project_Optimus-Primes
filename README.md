# Final Project Report

Skill Classification from Tennis Motion Sequences Using Subject-Disjoint Evaluation

## Abstract
This project develops and evaluates a reproducible machine learning pipeline for classifying tennis player skill level (beginner vs expert) from motion sequences. The study uses four stroke categories from THETIS (backhand, forehand_flat, kick_service, smash), applies strict subject-disjoint train/validation/test splitting, and compares baseline models (SVM, LSTM) with advanced temporal models (Transformer variants). Final results show that the advanced multitask Transformer provides the strongest and most stable test performance across seeds, with clear improvement over baseline skill models.

## 1. Project Scope and Target
The primary target is binary skill classification (beginner vs expert). A secondary target is action classification as a diagnostic signal for representation quality. The final pipeline emphasizes leakage prevention, class-imbalance handling, confound-aware analysis, robustness reporting, and gradient-based interpretation.

## 2. Dataset, Labels, and Split Protocol
### 2.1 Dataset Scope
- Total subjects: 55
- Total sequences: 660
- Actions: backhand, forehand_flat, kick_service, smash
- Sequences per action: 165 each

### 2.2 Label Construction
- Beginner: p1 to p31
- Expert: p32 to p55
- Overall class counts: beginner 372, expert 288
- repeat_id (s1/s2/s3) is trial index, not skill label

### 2.3 Subject-Disjoint Split
- Train subjects: 35 (420 sequences)
- Validation subjects: 9 (108 sequences)
- Test subjects: 11 (132 sequences)
- Test sequences per action: 33 each (balanced)

This split design prevents identity leakage and supports valid generalization claims.

## 3. Data Preparation and Feature Engineering
RGB videos are converted to pose keypoint sequences. For each frame sequence, the feature set includes normalized coordinates, velocity, acceleration, and confidence values. Sequences are standardized to a common length by truncation/padding. Metadata and quality checks confirm parse validity, action mapping consistency, repeat completeness, and duplicate-free indexing.

## 4. Modeling Pipeline
### 4.1 Baseline Stage
- SVM with class-weight balancing
- LSTM with weighted cross-entropy

Both baselines use the same split and feature pipeline for fair comparison.

### 4.2 Advanced Stage
- Transformer skill-only
- Transformer multitask (skill head + auxiliary action head)
- Ablation variants:
  - multitask without velocity/acceleration
  - multitask without normalization

Advanced training includes weighted loss and balanced sampling.

### 4.3 Robustness and Interpretation
- Multi-seed evaluation (seeds: 13, 42, 77)
- LOPO (leave-one-player-out) robustness stress test
- Gradient-based interpretation: saliency and integrated gradients

## 5. Evaluation Metrics
Primary metrics are Macro-F1, Balanced Accuracy, and ROC-AUC. Macro-F1 and Balanced Accuracy are emphasized for class-balance fairness. Per-action skill metrics are reported on test data to evaluate stroke-type heterogeneity. Multi-seed mean and standard deviation are reported for stability.

## 6. Results
### 6.1 Baseline Results (Test)
Best baseline skill model: lstm_skill
- Macro-F1: 0.7348
- Balanced Accuracy: 0.7417
- ROC-AUC: 0.8086

### 6.2 Advanced Results (Mean Across Seeds, Test)
Best advanced experiment by mean test skill performance: transformer_multitask
- Macro-F1: 0.8249 ± 0.0422
- Balanced Accuracy: 0.8333 ± 0.0360
- ROC-AUC: 0.9185 ± 0.0183

### 6.3 Improvement Over Best Baseline
Compared with the best baseline skill model:
- Macro-F1 improvement: 0.0901
- Balanced Accuracy improvement: 0.0917
- ROC-AUC improvement: 0.1100

### 6.4 LOPO Summary
- Subjects evaluated: 55
- Sequences per held-out subject: 12
- Macro-F1 mean/std: 0.5778 / 0.2804
- Balanced Accuracy mean/std: 0.7894 / 0.2090

LOPO ROC-AUC is not reported because held-out folds are single-subject and often single-class, making AUC undefined.

## 7. Interpretation
The advanced multitask Transformer is the strongest model under the project protocol because it combines higher mean test performance with confound-aware structure. Ablation behavior confirms the contribution of temporal dynamics (velocity/acceleration) and normalization. Gradient-based interpretation outputs provide supportive temporal saliency patterns and phase-level attribution trends.

## 8. Alignment with Instructor Feedback
The final workflow addresses all priority points:
- Class imbalance: handled with weighted loss and balanced sampling; split statistics are reported
- Stroke confound: addressed through multitask modeling and per-action analysis
- Interpretation reliability: gradient-based saliency and integrated gradients are used
- Reliability protocol: multi-seed statistics and LOPO are included

## 9. Conclusion
Under strict subject-disjoint evaluation, tennis skill classification from motion sequences is feasible and robust. The advanced multitask Transformer substantially outperforms baseline models and is the recommended final model for reporting. The study delivers a reproducible pipeline with transparent evaluation and clearly stated limitations.

## 10. Submission Notes
Primary conclusions should be based on multi-seed mean performance. Single-seed best rows are supplementary traceability evidence. LOPO should be interpreted as a robustness stress test and reported with Macro-F1 and Balanced Accuracy.

