train:
	python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 100 --tied --data data/lyrics

generate:
	python generate.py --cuda --data data/lyrics
	cat generated.txt

traincontrolgroup:
	python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 100 --tied


generatestdcontrolgroup:
	python generate.py --cuda
	cat generated.txt
