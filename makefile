device=0
lr=0.1
alpha=1.0
neg_samples=1

train-classification:
	python sse_train.py --task_type=classification --data_dir=rawdata-classification --model_dir=models-classification  --device=$(device) --learning_rate=$(lr) --alpha=$(alpha) --neg_samples=$(neg_samples) --max_epoc=50 --steps_per_checkpoint=200

index-classification:
	python sse_index.py  --idx_model_dir=models-classification --idx_rawfilename=targetIDs  --idx_encodedIndexFile=targetEncodingIndex.tsv

demo-classification:
	python sse_demo.py   --model_dir=models-classification --indexFile=targetEncodingIndex.tsv

train-qna:
	python sse_train.py --task_type=qna --data_dir=rawdata-qna --model_dir=models-qna  --batch_size=32 --max_epoc=200 --steps_per_checkpoint=10 --device=$(device) --learning_rate=$(lr) --vocab_size=8000 --max_seq_length=1000 --alpha=$(alpha) --neg_samples=$(neg_samples)

demo-qna:
	python sse_demo.py    --model_dir=models-qna --indexFile=targetEncodingIndex.tsv

index-qna:
	python sse_index.py  --idx_model_dir=models-qna --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

train-ranking:
	python sse_train.py --task_type=ranking --data_dir=rawdata-ranking --model_dir=models-ranking  --device=$(device) --learning_rate=$(lr) --alpha=$(alpha) --embedding_size=30 --encoding_size=50 --max_seq_length=50  --batch_size=32 --max_epoc=200 --steps_per_checkpoint=200  --neg_samples=$(neg_samples)

index-ranking:
	python sse_index.py  --idx_model_dir=models-ranking --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

demo-ranking:
	python sse_demo.py   --model_dir=models-ranking --indexFile=targetEncodingIndex.tsv

train-crosslingual:
	python sse_train.py --task_type=crosslingual --data_dir=rawdata-crosslingual --model_dir=models-crosslingual  --device=$(device) --learning_rate=$(lr) --alpha=$(alpha) --embedding_size=40 --encoding_size=50 --max_seq_length=50  --batch_size=32 --max_epoc=800 --steps_per_checkpoint=200  --neg_samples=$(neg_samples)


index-crosslingual:
	python sse_index.py  --idx_model_dir=models-crosslingual --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

demo-crosslingual:
	python sse_demo.py   --model_dir=models-crosslingual --indexFile=targetEncodingIndex.tsv


clean:
	rm *.log
	rm -rf models*
	rm *.pyc
	rm -rf __pycache__
