### 1.1 Data preprocessing:
* **show_json_data**: View the source data.

`python data_processor.py --phase show_json_data`

* **extract_abs_label**: Extract input and output from the source data. Because I use the aapr dataset, here are abs and label. For different source data, you need to write different data processing methods.

`python data_processor.py --phase extract_abs_label`

* **save_abs_label**: Save the processed clean data.

`python data_processor.py --phase save_abs_label`

* **split_data**: Split the clean data according to a certain ratio.

`python data_processor.py --phase split_data`

* **get_vocab**: Generate a dictionary for the deep learning part.

`python data_processor.py --phase get_vocab`

### 1.2 Training deep learning model:

* Training fully connected neural network

`python main.py --phase aapr.dl.mlp.norm`

* Training convolutional neural network

`python main.py --phase aapr.dl.textcnn.norm`