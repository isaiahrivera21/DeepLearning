from transformers import AutoTokenizer
from datasets import load_dataset
import tensorflow as tf 
from transformers import TFAutoModelForSequenceClassification


#load the AG New Dataset 
dataset = load_dataset('ag_news')

#both are list 
ag_labels = dataset['train']['label']
ag_data = dataset['train']['text']
sentences = [ag_data[0],ag_data[1],ag_data[3],ag_data[4]]


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(sentences, return_tensors="tf", padding=True)
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")


# The following display but it all it means is that we have to train the model 
# Some weights or buffers of the TF 2.0 model TFBertForSequenceClassification were
#  not initialized from the PyTorch model and are newly initialized:['classifier.weight', 'classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



#Now we need to train our model
#Maybe we can just use a linear model 


