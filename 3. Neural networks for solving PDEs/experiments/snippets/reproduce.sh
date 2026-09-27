python3 -m pip install -r experiments/requirements.txt
python3 experiments/forward_1d.py
python3 experiments/poisson_2d.py
python3 experiments/inverse_1d.py
python3 experiments/classical_accuracy.py
python3 experiments/make_figures.py
latexmk -pdf 3_neural_networks_for_pdes.tex
