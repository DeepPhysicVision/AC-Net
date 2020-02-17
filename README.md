# MHAC
code for the paper of 'Aspect-based Sentiment Analysis Based on Hybrid Multi-Head Attention and Capsule Networks'
CCL 2019
how to run:
1. run the xml_to_txt.py to read the original xml corpus file
2. run txt_to_input.py to chansfer the txt data to the model input data
3. run EmbeddingWriter.py to produce the embedding matrix, the pre_trained glove.840B.300d.txt is used in this part
4. run the main.py to train the model, it involve the model.py , Evaluator.py and ExampleReader.py
5. the reference paper 
