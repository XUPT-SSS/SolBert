
#### Introduction
SolBert 


#### Usage

1.  Install the required packages 
    Run the following commands: pip install -r requirements.txt
2.  Download the corresponding dataset：
	Link to experimental dataset:：https://drive.google.com/drive/folders/1vauZV2VbTkPSzDG9MM6E1O06Iaad_3Rl?usp=drive_link
    Experimentally constructed datasets: clone, bug and cluster datasets are placed under code/experiment/clone(bug|cluster)/data, respectively.
    
3.  Training model and parameters:
	The pre-training model  can be downloaded at: https://drive.google.com/drive/folders/1ZXg2r2xideI9w3bSeMicH4X2Gon714Bo?usp=drive_link
	pre-training model：solBert(code/solidity_model)
	mirror_soliBert: (code/mirror_bert/tmp/mirror_bert_mean/)
	bert_whitening:(code/bertWhitening/data/)
	mirror_bert_whitening:(code/mirror_bert_whitening/data/)

4. 	Experiment
	 1.modify code/config.py path  
	 2.modify  code/experiment/config.py path
	##### Clone
	  	step1 Build clone experimental dataset
		step2 to step5 use different methods respectively:
            python step2_bert_clone_detection.py avg_first_last (Clone experiment with avg_first_last pooling, the rest of the steps are similar)
    ##### Cluster
		Similar to the clone experiment
    ##### Bug 
		Similar to the clone experiment

      
