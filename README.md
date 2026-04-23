# Divide-and-Conquer: Prompt-based Distribution Learning for Multimodal  Sentiment Analysis

This is the code repository for the paper:   "Divide-and-Conquer: Prompt-based Distribution Learning for Multimodal Sentiment Analysis"  

## Note

The code has been refactored. Please raise any issues.

## Usage

1. Prepare the dataset  and pre-trained RoBERTa.
   For dataset , you can download the processed data from [Self-MM](https://github.com/thuiar/Self-MM).
   For RoBERTa model, you can download [robert-large ](https://huggingface.co/FacebookAI/roberta-large)and [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) from [Hugging Face](https://huggingface.co/).

2. Clone this repo and install requirements.

   ```shell
   git clone https://github.com/Dongri404/PDL.git
   cd PDL
   conda create --name pdl python=3.8
   source activate pdl
   pip install -r requirements.txt
   ```

3. Update path.
   Modify the `config/config_tune.py` and `config/config_regression.py` to update dataset paths . Update the pretrained model path in `models/subNets/BertTextEncoder.py`.

4. Run

   ```shell
   python -u run.py --datasetNames sims mosi mosei
   ```
