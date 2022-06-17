cd ../
python source/evaluate.py \
--weights result/version_0004/best.pt \
--logfile result/version_0004/result.txt \
--cfg config/version_004/train_config.yaml \
--data config/version_004/data_config.yaml \
--batch_size 512 \
--device cuda:4
