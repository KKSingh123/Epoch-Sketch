# UNI Experiment Files

## Code
- `code/uni_prf_metrics.py`
- `code/plot_uni_prf_results.py`

## Results
- `results/uni_prf_results.json`
- `results/uni_precision.pdf`
- `results/uni_recall.pdf`
- `results/uni_f1_score.pdf`

## Commands
- Generate precision/recall/F1 data: `python3 uni/code/uni_prf_metrics.py`
- Plot precision/recall/F1 data: `python3 uni/code/plot_uni_prf_results.py`


## Command (For caida data set ARE)
- cd caida/results && PYTHONPATH=../.. python3 ../code/memory_experiment.py --dat ../../dataset/CAIDA/0.dat && cd ../..

<!-- For F1, Precision and recall Results -->
- python3 caida/code/caida_prf_metrics.py --dat dataset/CAIDA/0.dat --output caida/results/caida_prf_results.json


<!-- For Presion, Recall and F1 score (Plots) -->
- python3 caida/code/plot_caida_prf_results.py --input caida/results/caida_prf_results.json --output-dir caida/results

<!-- For ARE, Top ARE and throughput (Plots) -->
- cd caida/results && python3 ../code/plot_saved_results.py --input sketch_results.json && cd ../..


## Command (For Uni data set ARE)

- cd uni/results && PYTHONPATH=../.. python3 ../code/uni_memory_experiment.py --csv ../../dataset/UNI/univ1_pt0.csv --output uni_sketch_results.json --plot-prefix uni_ && cd ../..

<!-- For F1, Precision and recall Results -->
- python3 uni/code/uni_prf_metrics.py --csv dataset/UNI/univ1_pt0.csv --output uni/results/uni_prf_results.json

<!-- For Presion, Recall and F1 score (Plots) -->
- python3 uni/code/plot_uni_prf_results.py --input uni/results/uni_prf_results.json --output-dir uni/results

<!-- For ARE, Top ARE and throughput (Plots) -->
- cd uni/results && python3 ../code/plot_uni_saved_results.py --input uni_sketch_results.json && cd ../..

























#