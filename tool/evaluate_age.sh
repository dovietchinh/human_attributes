python source/evaluate_age.py \
--weights ./result/runs_human_attributes_age_5/best.pt \
--logfile result/runs_human_attributes_age_5/result.txt \
--cfg config/human_attribute_age_5/train_config.yaml \
--data config/human_attribute_age_5/data_config.yaml \
--batch_size 512 \
--device cuda:5