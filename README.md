# LRCount-CNN
This code is for our ICML 2020 paper "On the Number of Linear Regions of Convolutional Neural Networks."


requirement: install python, pytorch and numpy 


1. Exp1: One-Layer CNN

    The python script is 'RandInput_LRCount.py' and one can run the experiments by "python RandInput_LRCount.py".  One can change the parameters (e.g., the out_channel number, the input/kernel size) to validate the results shown in the submitted paper. 
       
       We also provide a bash script 'execute_OneLayerCNN.sh' in the 'experiments' directory for running the experiments with multiple random seeds and standard deviations (by "bash execute_OneLayerCNN.sh")


 2. Exp2: Two-Layer CNN

            The python script is 'RandInput_LRCount_TwoLayer.py' and one can run the experiments by "python RandInput_LRCount_TwoLayer.py".  One can change the parameters (e.g., the out_channel number) to validate the results shown in the submitted paper. 
                
                We also provide a bash script 'execute_TwoLayerCNN.sh' in the 'experiments' directory for running the experiments with multiple random seeds and standard deviations (by "bash execute_TwoLayerCNN.sh")

