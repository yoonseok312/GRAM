# GRAM [NAACL 2022 Main, Oral]

Official PyTorch Implementation of [GRAM: Fast Fine-tuning of Pre-trained Language Models for Content-based Collaborative Filtering](https://arxiv.org/abs/2204.04179).
Along with Knowledge Tracing and News Recommendation codebase, we provide a simple MNIST example to play around with GRAM. 

Below `gram` folder, you can find respective models for knowledge tracing and news recommendation in `knowledge_tracing` and `news_recommendation` repo. 
To run models, use `run_script.py` for knowledge tracing and `run_pl_nrms.py` for news recommendation. Before running the script, we recommend installing the gram package with:
```bash
pip install -r requirements.txt
pip install -e gram
```

Datasets can be found [here](https://tinyurl.com/gram-datasets).

If you find this repo useful, please cite our work! The proceedings bibtex will be available soon. 

