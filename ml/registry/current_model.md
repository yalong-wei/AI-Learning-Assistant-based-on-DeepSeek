# Current Production Model

- Name: intent-classifier
- Stage: Production
- URI: models:/intent-classifier/Production
- Selection rationale: higher macro-F1 vs baseline; verified via MLflow artifacts (confusion_matrix.csv, classification_report.json)
- Dataset version: aligned with DVC v2 (to be updated with actual DVC hash)
- Code version: see MLflow tag `git_sha` in the Production run
