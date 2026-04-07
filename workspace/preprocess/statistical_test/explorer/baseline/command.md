
```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_2_v1/lgbm_classifier.joblib \
  --metrics-json workspace/model/artifacts/lgbm_2_v1/metrics.json \
  --threshold 0.5  \
  --eval-parquet workspace/dataset/unpreprocessed/test/holdout_train.parquet \
  --eval-parquet workspace/dataset/unpreprocessed/test/holdout_test.parquet \
  --eval-parquet workspace/dataset/unpreprocessed/test/pb_human_bot.parquet \
  --eval-parquet workspace/dataset/unpreprocessed/test/pb_human_bot_val.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/baseline/cross_eval
  
```