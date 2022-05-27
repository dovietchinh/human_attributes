python source/evaluate.py \
--weights ./result/runs_human_attributes_24/best.pt \
--logfile result/runs_human_attributes_24/result.txt \
--cfg config/human_attribute_24/train_config.yaml \
--data config/human_attribute_24/data_config.yaml \
--batch_size 512 \
--device cuda:3