# CDR2IMG
TensorFlow implementation for the paper below:

**DR2IMG: A Bridge from Text to Image in Telecommunication Fraud Detection**

## Running CDR2IMG

1.To run the code, you need to install the dependencies descirbed in env_list.txt.

2.Go to [this site](https://aistudio.baidu.com/aistudio/datasetdetail/40690) to download
the datasets, namely train_app.csv,train_sms.csv,train_user.csv,train_voc.csv. We didn't upload it because it's too large, and couldn't upload to github.

3.Unzip /CDR2IMG/data.zip and put the 4 downloaded datasets in the path: /CDR2IMG/data/

4.Run CDR2IMG/src/pre_process/produce_2d_behavior_matrix_hour.py to generate image-like matrixes.

5.Run CDR2IMG/src/models/CNNs/self-built_cnn_without_augmentation.py to run CDR2IMG.
