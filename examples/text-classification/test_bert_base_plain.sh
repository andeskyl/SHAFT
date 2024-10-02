export TASK_NAME=sst2

python run_glue_eval.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_eval_batch_size 8 \
  --output_dir eval/$TASK_NAME/
