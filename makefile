device=0
lr=0.1

train-classification:
	python sse_main.py --task_type=classification --data_dir=rawdata-classification --model_dir=models-classification  --device=$(device) --learning_rate=$(lr)

demo-classification:
	python sse_main.py --demo=True --task_type=classification  --model_dir=models-classification

train-qna:
	python sse_main.py --task_type=qna --data_dir=rawdata-qna --model_dir=models-qna  --batch_size=16 --max_epoc=1000 --steps_per_checkpoint=10 --device=$(device) --learning_rate=$(lr) --vocab_size=8000 --max_seq_length=1000

demo-qna:
	python sse_main.py --demo=True --task_type=qna  --model_dir=models-qna

train-ranking:
	echo "Search ranking task will be supported very soon."

demo-ranking:
	echo "Search ranking task will be supported very soon."

train-crosslingual:
	echo "Cross-Lingual information retrieval task will be supported very soon."

demo-crosslingual:
	echo "Cross-Lingual information retrieval task will be supported very soon."


clean:
	rm *.log
	rm -rf models*
	rm *.pyc
	rm -rf __pycache__
