install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval-lax ./3_regressions/*.ipynb


all: install                                              test