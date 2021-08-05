# WestBERT
This is code for Accelerating Pretrained Language Model Inference Using Weighted Ensemble Self-Distillation
 
## Requirements

```
transformers == 4.2.1  
torch == 1.7.1  
python == 3.8.5  
tqdm == 4.50.2  
```
Download GLUE dataset by
```
python download_glue_data.py --data_dir data --tasks all
```

##Usage
### Training

You can fine-tune the WestBERT model by:
```
TASK_NAME = SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
GLUE_DIR = ./glue_data

python ./run_AdaEE_glue.py 
        --model_type bert 
        --model_name_or_path bert-base-uncased
        --task_name $TASK_NAME 
        --do_train
        --do_eval
        --do_lower_case 
        --data_dir $GLUE_DIR/$TASK_NAME
        --max_seq_length 128 
        --per_gpu_train_batch_size 128
        --per_gpu_eval_batch_size 1 
        --learning_rate 2e-5 
        --logging_steps 50 
        --save_steps 50
        --seed 42
        --overwrite_output_dir
        --num_train_epochs 15 
        --output_dir ./save/ 
```
### Inference

You can inference with different confidence threshold settings by:
```
TASK_NAME = SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
GLUE_DIR = ./glue_data
ENTROPIES=0.2

python ./run_AdaEE_glue.py 
        --model_type bert
        --model_name_or_path bert-base-uncased
        --task_name $TASK_NAME
        --do_eval 
        --do_lower_case 
        --data_dir $GLUE_DIR/$TASK_NAME
        --output_dir ./save/
        --max_seq_length 128 
        --early_exit_entropy $ENTROPY
        --eval_threshold
        --per_gpu_eval_batch_size=1 
        --eval_all_checkpoints
```

 