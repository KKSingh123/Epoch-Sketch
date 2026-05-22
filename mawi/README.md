# MAWI Experiment Files

## Code
- `code/mawi_demo.py`
- `code/mawi_memory_experiment.py`
- `code/mawi_prf_metrics.py`
- `code/plot_mawi_saved_results.py`
- `code/plot_mawi_prf_results.py`
- `code/mawi26_demo.py`
- `code/mawi26_memory_experiment.py`
- `code/mawi26_prf_metrics.py`
- `code/plot_mawi26_saved_results.py`
- `code/plot_mawi26_prf_results.py`

## Results
- `results/sketch_results.json`
- `results/mawi_prf_results.json`
- `results/mawi_throughput.pdf`
- `results/mawi_are_top20.pdf`
- `results/mawi_are_all.pdf`
- `results/mawi_are_hh.pdf`
- `results/mawi_precision.pdf`
- `results/mawi_recall.pdf`
- `results/mawi_f1_score.pdf`
- `results/mawi26_sketch_results.json`
- `results/mawi26_prf_results.json`
- `results/mawi26_throughput.pdf`
- `results/mawi26_are_top20.pdf`
- `results/mawi26_are_all.pdf`
- `results/mawi26_are_hh.pdf`
- `results/mawi26_precision.pdf`
- `results/mawi26_recall.pdf`
- `results/mawi26_f1_score.pdf`

## Commands
- Generate throughput/ARE data:
  `cd mawi/results && PYTHONPATH=../.. python3 ../code/mawi_memory_experiment.py --dat ../../dataset/Mawi/mawi.dat && cd ../..`
- Plot saved throughput/ARE data:
  `cd mawi/results && python3 ../code/plot_mawi_saved_results.py --input sketch_results.json && cd ../..`
- Generate precision/recall/F1 data:
  `python3 mawi/code/mawi_prf_metrics.py --dat dataset/Mawi/mawi.dat --output mawi/results/mawi_prf_results.json`
- Plot saved precision/recall/F1 data:
  `python3 mawi/code/plot_mawi_prf_results.py --input mawi/results/mawi_prf_results.json --output-dir mawi/results`

## Plotted Comparison
- `CountLess`
- `Elastic`
- `Stable`
- `HFH Sketch`




## Command (For MAWI data set ARE)
- cd mawi/results && PYTHONPATH=../.. python3 ../code/mawi_memory_experiment.py --dat ../../dataset/Mawi/mawi.dat && cd ../..

<!-- For F1, Precision and recall Results -->
- python3 mawi/code/mawi_prf_metrics.py --dat dataset/Mawi/mawi.dat --output mawi/results/mawi_prf_results.json

<!-- For Precision, Recall and F1 score (Plots) -->
- python3 mawi/code/plot_mawi_prf_results.py --input mawi/results/mawi_prf_results.json --output-dir mawi/results

<!-- For ARE, Top ARE and throughput (Plots) -->
- cd mawi/results && python3 ../code/plot_mawi_saved_results.py --input sketch_results.json && cd ../..

## Command (For MAWI26 data set ARE)
- cd mawi/results && PYTHONPATH=../.. python3 ../code/mawi26_memory_experiment.py --dat ../../dataset/Mawi/mawi26.dat && cd ../..

<!-- For F1, Precision and recall Results -->
- python3 mawi/code/mawi26_prf_metrics.py --dat dataset/Mawi/mawi26.dat --output mawi/results/mawi26_prf_results.json

<!-- For Precision, Recall and F1 score (Plots) -->
- python3 mawi/code/plot_mawi26_prf_results.py --input mawi/results/mawi26_prf_results.json --output-dir mawi/results

<!-- For ARE, Top ARE and throughput (Plots) -->
- cd mawi/results && python3 ../code/plot_mawi26_saved_results.py --input mawi26_sketch_results.json && cd ../..



Extract data from pcap file command
- python3 Downloads/pcap_to_mawi_dat.py --input Downloads/202601011400.pcap.gz --output mawi26.dat --count 8000000
