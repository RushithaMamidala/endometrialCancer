# endometrialCancer
To extract the text from American Cancer Society recommendation
```
python guidelineExtractors/acs.py
```

To run an evaluation pass: 
```
MODEL=openai/gpt-oss-20b

python3 -u run_clm.py --model_name_or_path ${MODEL} --train_file guidelineExtractors/guidelines/acs.json --output_dir checkpoints/${MODEL} --do_eval --eval_subset train
```

To build a datstore with FAISS indexing for the extracted text run
```
MODEL=openai/gpt-oss-20b

python3 -u run_clm.py --model_name_or_path ${MODEL} --train_file guidelineExtractors/guidelines/acs.json --output_dir checkpoints/${MODEL} --do_eval --eval_subset train --dstore_dir checkpoints/${MODEL} --build_index
```
Make a note of the datastore size from this step.
To cluster the datastore run 
```
MODEL=openai/gpt-oss-20b

python3 -u run_clm.py --model_name_or_path ${MODEL} --train_file guidelineExtractors/guidelines/acs.json --output_dir checkpoints/${MODEL} --eval_subset train --dstore_dir checkpoints/${MODEL} --dstore_size 15359 --cluster_dstore --num_clusters 1500 --sample_size 15359 
```

To perform the knn evaluation run 
```
MODEL=openai/gpt-oss-20b

python3 -u run_clm.py --model_name_or_path ${MODEL} --train_file guidelineExtractors/guidelines/acs.json --output_dir checkpoints/${MODEL} --do_eval --eval_subset train --dstore_dir checkpoints/${MODEL} --dstore_size 15359 --knn --k 1024 --knn_temp 1 --lmbda 0.1
```

To run evaluation with the retoMaton
```
MODEL=openai/gpt-oss-20b

python3 -u run_clm.py --model_name_or_path ${MODEL} --train_file guidelineExtractors/guidelines/acs.json --output_dir checkpoints/${MODEL} --do_eval --eval_subset train --dstore_dir checkpoints/${MODEL} --dstore_size 127999 --retomaton --k 1024 --knn_temp 1 --lmbda 0.1
```

Generate recommnedation
```
MODEL=openai/gpt-oss-20b

python3 -u run_recomm.py --model_name_or_path ${MODEL} --train_file path_reports_dataset.csv --output_dir recommendations/ --eval_subset train --do_gen --dstore_dir checkpoints/${MODEL} --dstore_size 15359 --retomaton --lmbda 0.1
```

Extract Key Features
```
MODEL=openai/gpt-oss-20b

python3 -u run_recomm.py --model_name_or_path ${MODEL} --train_file path_reports_dataset.csv --output_dir recommendations/ --eval_subset train --do_gen --dstore_dir checkpoints/${MODEL} --dstore_size 15359 --retomaton --lmbda 0.1 --report
```

Dstore with basecases - 118783
