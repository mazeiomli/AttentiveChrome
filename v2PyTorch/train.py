import warnings
warnings.filterwarnings("ignore")
import argparse
import json
# import matplotlib
# import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import models as Model
# from SiameseLoss import ContrastiveLoss
import evaluate
import data
import gc
import csv
from pdb import set_trace as stop

from timeit import default_timer as timer
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
print('Starting at time: ', dt_string)

# python train.py --cell_type=Cell1 --model_name=attchrome --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/

parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', type=str, default='attchrome', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--cell_type', type=str, default='E003', help='cell type 1')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--data_root', type=str, default='./data/', help='data location')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_on_saved_model',action='store_true', help='only test on saved model')
args = parser.parse_args()




torch.manual_seed(1)

# set model_name
model_name = ''
model_name += (args.cell_type)+('_')

model_name+=args.model_type



args.bidirectional=not args.unidirectional

print('the model name: ',model_name)
args.data_root+=''
args.save_root+=''
args.dataset=args.cell_type
args.data_root = os.path.join(args.data_root)
print('loading data from:  ',args.data_root)
args.save_root = os.path.join(args.save_root,args.dataset)
print('saving results in  from: ',args.save_root)
model_dir = os.path.join(args.save_root,model_name)
if not os.path.exists(model_dir):
    # when exist_ok=False, errors if target directory already exists
    # doesn't error if intermediate dirs exist
    # here, exist_ok=False is fine since we check if
    # model_dir exists first
    os.makedirs(model_dir)



attentionmapfile=model_dir+'/'+args.attentionfilename
print('==>processing data')
Train,Valid,Test = data.load_data(args)






print('==>building model')
model = Model.att_chrome(args)



if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(1)
    dtype = torch.cuda.FloatTensor
    # cuda.set_device(args.gpuid)
    model.type(dtype)
    print('Using GPU '+str(args.gpuid))
else:
    dtype = torch.FloatTensor

print(model)
if(args.test_on_saved_model==False):
    print("==>initializing a new model")
    for p in model.parameters():
        p.data.uniform_(-0.1,0.1)


optimizer = optim.Adam(model.parameters(), lr = args.lr)
#optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
def train(TrainData):
    model.train()
    # initialize attention
    diff_targets = torch.zeros(TrainData.dataset.__len__(),1)
    predictions = torch.zeros(diff_targets.size(0),1)

    all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
    all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)

    num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
    all_gene_ids=[None]*TrainData.dataset.__len__()
    per_epoch_loss = 0
    print('Training')
    for idx, Sample in enumerate(TrainData):

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())
    

        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()

        
        optimizer.zero_grad()
        batch_predictions= model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')

        per_epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # all_attention_bin[start:end]=batch_alpha.data
        # all_attention_hm[start:end]=batch_beta.data

        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()
        
    per_epoch_loss=per_epoch_loss/num_batches
    return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids



def test(ValidData,split_name):
    model.eval()

    diff_targets = torch.zeros(ValidData.dataset.__len__(),1)
    predictions = torch.zeros(diff_targets.size(0),1)

    all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
    all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

    num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
    all_gene_ids=[None]*ValidData.dataset.__len__()
    per_epoch_loss = 0
    print(split_name)
    for idx, Sample in enumerate(ValidData):

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
        optimizer.zero_grad()

        inputs_1 = Sample['input']
        batch_diff_targets= Sample['label'].unsqueeze(1).float()
        

        # batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
        batch_predictions = model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')
        # all_attention_bin[start:end]=batch_alpha.data
        # all_attention_hm[start:end]=batch_beta.data


        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()

        per_epoch_loss += loss.item()
    per_epoch_loss=per_epoch_loss/num_batches
    return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids



# best_valid_loss not used
best_valid_loss = 10000000000
# save valid and test metrics of best model wrt avgAUPR
best_valid_avgAUPR=-1
best_test_avgAUPR=-1
valid_metrics_best_valid_avgAUPR = []
test_metrics_best_valid_avgAUPR = []
epoch_best_balid_avgAUPR = -1
# save valid and test metrics of best model wrt avgAUC
best_valid_avgAUC=-1
best_test_avgAUC=-1
valid_metrics_best_valid_avgAUC = []
test_metrics_best_valid_avgAUC = []
epoch_best_valid_avgAUC = -1
if(args.test_on_saved_model==False):
    # measure training time
    train_start_time = timer()
    # train model and save best one
    for epoch in range(0, args.epochs):
        print('---------------------------------------- Training '+str(epoch+1)+' -----------------------------------')
        predictions,diff_targets,alpha_train,beta_train,train_loss,_ = train(Train)
        # Future work: refactor how the return values are stored to make
        # accessing, printing with metric names, and manipulation easier
        train_avgAUPR, train_medAUPR, train_varAUPR, train_avgAUC, train_medAUC, train_varAUC = evaluate.compute_metrics(predictions,diff_targets)

        predictions,diff_targets,alpha_valid,beta_valid,valid_loss,gene_ids_valid = test(Valid,"Validation")
        valid_avgAUPR, valid_medAUPR, valid_varAUPR, valid_avgAUC, valid_medAUC, valid_varAUC = evaluate.compute_metrics(predictions,diff_targets)

        predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test,'Testing')
        test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC = evaluate.compute_metrics(predictions,diff_targets)

        # if valid_avgAUC improved, save model checkpoint and stats
        if(valid_avgAUC >= best_valid_avgAUC):
            # save best epoch -- models converge early
            print('Saving new best validation avgAUC model at epoch ', epoch, '...')
            best_valid_avgAUC = valid_avgAUC
            best_test_avgAUC = test_avgAUC
            epoch_best_valid_avgAUC = epoch+1
            valid_metrics_best_valid_avgAUC = [valid_avgAUPR, valid_medAUPR, valid_varAUPR, valid_avgAUC, valid_medAUC, valid_varAUC]
            test_metrics_best_valid_avgAUC = [test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC]
            torch.save(model.cpu().state_dict(),model_dir+"/"+model_name+'_avgAUC_model.pt')
            model.type(dtype)

        # if valid_avgAUPR improved, save model checkpoint and stats
        if(valid_avgAUPR >= best_valid_avgAUPR):
            # save best epoch -- models converge early
            print('Saving new best validation avgAUPR model at epoch ', epoch, '...')
            best_valid_avgAUPR = valid_avgAUPR
            best_test_avgAUPR = test_avgAUPR
            epoch_best_balid_avgAUPR = epoch+1
            valid_metrics_best_valid_avgAUPR = [valid_avgAUPR, valid_medAUPR, valid_varAUPR, valid_avgAUC, valid_medAUC, valid_varAUC]
            test_metrics_best_valid_avgAUPR = [test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC]
            torch.save(model.cpu().state_dict(),model_dir+"/"+model_name+'_avgAUPR_model.pt')
            model.type(dtype)

        print("Epoch:",epoch)
        # train, valid, test AUPR metrics for the epoch
        print("train avgAUPR:",train_avgAUPR)
        print("valid avgAUPR:",valid_avgAUPR)
        print("test avgAUPR:",test_avgAUPR)
        # metrics for best avgAUPR model
        print("best valid avgAUPR:", best_valid_avgAUPR)
        print("best test avgAUPR:", best_test_avgAUPR)
        # train, valid, test AUC metrics for the epoch
        print("train avgAUC:",train_avgAUC)
        print("valid avgAUC:",valid_avgAUC)
        print("test avgAUC:",test_avgAUC)
        # metrics for best avgAUC model
        print("best valid avgAUC:", best_valid_avgAUC)
        print("best test avgAUC:", best_test_avgAUC)
        # flush output each iteration to see live updates
        sys.stdout.flush()

    train_end_time = timer()
    # training time in minutes
    train_elapsed_time = round((train_end_time - train_start_time) / (60), 3)
    print("\nFinished training")
    print("Training time (min): ", train_elapsed_time)
    # print metrics for best valid avgAUPR model
    print("Best epoch for valid avgAUPR: ", epoch_best_balid_avgAUPR)
    print("Best validation avgAUPR: ", best_valid_avgAUPR)
    print("Best test avgAUPR: ", best_test_avgAUPR)
    print("Copypaste metrics for best valid avgAUPR")
    print(','.join([str(x) for x in test_metrics_best_valid_avgAUPR] + [str(x) for x in valid_metrics_best_valid_avgAUPR]))

    # print metrics for best valid avgAUC model
    print("\nBest epoch for valid avgAUC: ", epoch_best_valid_avgAUC)
    print("Best validation avgAUC: ",best_valid_avgAUC)
    print("Best test avgAUC: ",best_test_avgAUC)
    # print("copypaste header: test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC, valid_avgAUPR, valid_medAUPR, valid_varAUPR, valid_avgAUC, valid_medAUC, valid_varAUC")
    print("Copypaste metrics for best valid avgAUC")
    print(','.join([str(x) for x in test_metrics_best_valid_avgAUC] + [str(x) for x in valid_metrics_best_valid_avgAUC]))


    if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
            gene_attention=[]
            gene_attention.append(gene_ids_test[i])
            for e in beta_test[i,:]:
                gene_attention.append(str(e))
            attentionfilewriter.writerow(gene_attention)
        attentionfile.close()


else:
    # load model and test
    model=torch.load(model_dir+"/"+model_name+'_avgAUC_model.pt')
    predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test)
    test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC = evaluate.compute_metrics(predictions,diff_targets)
    test_metrics = [test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC]
    print("\ntest avgAUC:",test_avgAUC)
    print("copypaste header: test_avgAUPR, test_medAUPR, test_varAUPR, test_avgAUC, test_medAUC, test_varAUC")
    print(','.join([str(x) for x in test_metrics]))

    if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
            gene_attention=[]
            gene_attention.append(gene_ids_test[i])
            for e in beta_test[i,:]:
                gene_attention.append(str(e))
            attentionfilewriter.writerow(gene_attention)
        attentionfile.close()

print('the model name: ',model_name)