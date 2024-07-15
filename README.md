# SPS-Bert

**SciBERT**:
   - SciBERT is a BERT model pre-trained on a large corpus of scientific text, making it particularly effective for tasks within the scientific domain.
   - Download: [GitHub - allenai/scibert](https://github.com/allenai/scibert)

## Data Location
Data files used in this project are stored in the `/data` directory. 
Due to the huge amount of data, please download the original data from the following link:
**DBLP**: 
   - Download: https://www.aminer.cn/citation
**PubMed**:
   - Download: https://pubmed.ncbi.nlm.nih.gov/download/

## Data processor
To process the data, use the `data_processor.py` script. This script takes the raw data files and prepares them for training, validation, and testing phases.
```bash
python data_processor.py --data dblp --data_name dblp.csv
```

## Training, Validation, and Testing
The following commands demonstrate how to train, validate, and test different variants of the SPS-BERT model using the provided scripts.

**Training SPS-BERT**
Train the SPS-BERT model in DI-5:
```bash
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d
```

Train the SPS-BERT model in DI-10:
```bash
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d_new
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d_new
```

Train the SPS-BERT model in citation:
```bash
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label num
python main.py --model spsbert --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label num
```

**Training ONlypro**
Train the OnlyPro model in DI-5:
```bash
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d
```

Train the OnlyPro model in DI-10:
```bash
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d_new
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d_new
```

Train the OnlyPro model in DI-10:
```bash
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label num
python main.py --model onlypro --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label num
```

**Training ONlyreference**
Train the Onlyreference model in DI-5:
```bash
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d
```

Train the Onlyreference model in DI-10:
```bash
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label d_new
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label d_new
```

Train the Onlyreference model in DI-5:
```bash
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data dblp --label num
python main.py --model onlyref --epochs 3 --lr 0.001 --batch_size 64 --data pmc --label num
```