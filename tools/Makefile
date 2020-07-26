#
# Makefile
# Masao Someki, 2019-01-30
#

.PHONY: all clean

all: venv gpu


venv:
	python3 -m venv venv
	. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

cpu:
	. venv/bin/activate; pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

gpu:
	. venv/bin/activate; pip install torch torchvision

clean:
	rm -fr venv dist build *.egg-info word2vec.*

# vim:ft=make
#
