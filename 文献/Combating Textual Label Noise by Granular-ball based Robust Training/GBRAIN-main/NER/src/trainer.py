from collections import Counter
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# BertPreTokenizer

from transformers import (AdamW, RobertaTokenizer, get_linear_schedule_with_warmup)
from tqdm import tqdm
from seqeval.metrics import classification_report
from utils import RoSTERUtils
from model import GBRAINModel

from torch.optim.lr_scheduler import MultiStepLR

from loss import GCELoss


from visdom import Visdom
viz = Visdom()
viz.line([0.],[0.],win="prec",opts=dict(title="prec"))
viz.line([0.], [0.], win="rec", opts=dict(title="rec"))
viz.line([0.], [0.], win="f1", opts=dict(title="f1"))
viz.line([1.], [0.], win="loss", opts=dict(title="train_loss"))



class RoSTERTrainer(object):

    def __init__(self, args):
        self.args = args
        
        self.seed = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        
        self.output_dir = args.output_dir
        self.data_dir = args.data_dir
        self.temp_dir = args.temp_dir
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        if args.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, must be >= 1")

        if args.train_batch_size != 32:
            print(f"Batch size for training is {args.train_batch_size}; 32 is recommended!")
            #exit(-1)
        self.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        self.eval_batch_size = args.eval_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_seq_length = args.max_seq_length
        
        self.self_train_update_interval = args.self_train_update_interval * args.gradient_accumulation_steps
        self.noise_train_update_interval = args.noise_train_update_interval * args.gradient_accumulation_steps
        self.noise_train_epochs = args.noise_train_epochs
        self.ensemble_train_epochs = args.ensemble_train_epochs
        self.self_train_epochs = args.self_train_epochs

        self.warmup_proportion = args.warmup_proportion
        self.weight_decay = args.weight_decay
        self.q = args.q
        self.tau = args.tau
        
        self.noise_train_lr = args.noise_train_lr
        self.ensemble_train_lr = args.ensemble_train_lr
        self.self_train_lr = args.self_train_lr
        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False)
        self.processor = RoSTERUtils(self.data_dir, self.tokenizer)
        self.label_map, self.inv_label_map = self.processor.get_label_map(args.tag_scheme)
        self.num_labels = len(self.inv_label_map) - 1  # exclude UNK type
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.mask_id = self.tokenizer.mask_token_id

        
        # Prepare model
        self.model = GBRAINModel.from_pretrained(args.pretrained_model, num_labels=self.num_labels-1,
                                                 hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"***** Using {torch.cuda.device_count()} GPU(s)! *****\n")
        if torch.cuda.device_count() > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        if args.do_train:
            tensor_data = self.processor.get_tensor(dataset_name="train", max_seq_length=self.max_seq_length, supervision='dist', drop_o_ratio=0.5)
            
            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.tensor_data = tensor_data
            self.gce_bin_weight = torch.ones_like(all_input_ids).float()
            self.gce_type_weight = torch.ones_like(all_input_ids).float()
            
            self.train_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)

            print("***** Training stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.train_batch_size}")

        if args.do_eval:
            tensor_data = self.processor.get_tensor(dataset_name=args.eval_on, max_seq_length=self.max_seq_length, supervision='true')

            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.y_true = tensor_data["raw_labels"]

            eval_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)
            eval_sampler = SequentialSampler(eval_data)
            self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            print("***** Evaluation stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.eval_batch_size}")

    # prepare model, optimizer and scheduler for training
    def prepare_train(self, lr, epochs):
        model = self.model.to(self.device)
        if self.multi_gpu:
            model = nn.DataParallel(model)
        num_train_steps = int(len(self.train_data)/self.train_batch_size/self.gradient_accumulation_steps) * epochs
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        
            
        optimizer_grouped_parameters = [
             {'params': model.roberta.parameters(), 'lr': 1e-6,'weight_decay': self.weight_decay},
            # {'params': model.bert.parameters(), 'lr': 1e-5,'weight_decay': self.weight_decay},
             {'params': model.fc1.parameters(), 'lr': 1e-4,'weight_decay': self.weight_decay},
             {'params': model.fc2.parameters(), 'lr': 1e-4,'weight_decay': self.weight_decay},
            #  {'params': model.ball_attention.parameters(), 'lr': 1e-3,'weight_decay': self.weight_decay},
             {'params': model.classifier.parameters(), 'lr': 1e-4,'weight_decay': self.weight_decay},
             {'params': model.bin_classifier.parameters(), 'lr': 1e-4,'weight_decay': self.weight_decay},
         
         ]    
        
        warmup_steps = int(self.warmup_proportion*num_train_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=self.weight_decay,eps=1e-8)
        
        # scheduler = MultiStepLR(optimizer, milestones=[int(20), int(30)], gamma=0.5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
        model.train()
        return model, optimizer, scheduler

    # training model on distantly-labeled data with noise-robust learning
    def noise_robust_train(self, model_idx=0):
        if os.path.exists(os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt")):
            print(f"\n\n******* Model {model_idx} predictions found; skip training *******\n\n")
            # return
            
        else:
            print(f"\n\n******* Training model {model_idx} *******\n\n")
        model, optimizer, scheduler = self.prepare_train(lr=self.noise_train_lr, epochs=self.noise_train_epochs)
        
        print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        loss_fct = GCELoss(0.7)
        
        i = 0
        
        if False:
          for epoch in range(1):
            bin_loss_sum = 0
            type_loss_sum = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                if step==200:
                    break
                model.training = True
                if step %100 ==0:
                  y_pred, _ = self.eval(model, self.eval_dataloader)
                  print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                  self.performance_report(self.y_true, y_pred)
                
                if (i+1) % self.noise_train_update_interval == 0:
                    #self.update_weights(model)
                    model.train()
                    
                    print(f"bin_loss: {round(bin_loss_sum/self.noise_train_update_interval,5)}; type_loss: {round(type_loss_sum/self.noise_train_update_interval,5)}")
                    bin_loss_sum = 0
                    type_loss_sum = 0
                idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
                
                bin_weights = self.gce_bin_weight[idx].to(self.device)
                type_weights = self.gce_type_weight[idx].to(self.device)

                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights))
                
                type_logits, bin_logits,labels,sb = model(input_ids, attention_mask, valid_pos,labels,flage=0)
                
                #labels = labels[valid_pos > 0]
                #bin_weights = bin_weights[valid_pos > 0]
                #type_weights = type_weights[valid_pos > 0]

                bin_labels = labels.clone()
                bin_labels[labels > 0] = 1
                type_labels = labels - 1
                type_labels[type_labels < 0] = -100
                
                

                type_loss = loss_fct(type_logits.view(-1, self.num_labels-1), type_labels.view(-1), type_weights).sum()
                
                # print(type_loss)
                type_loss_sum += type_loss.item()

                bin_loss = loss_fct(bin_logits.view(-1, 1), bin_labels.view(-1), bin_weights).sum()
                bin_loss_sum += bin_loss.item()
                
                loss = type_loss + bin_loss
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                i += 1
            
            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)

        
        x_scale = 0
        max_f1 = 0
        loss1 = 0
        loss2 = 0
        bestF1 = 0
        for epoch in range(self.noise_train_epochs):
            # if epoch == 1000:
            #     break

            bin_loss_sum = 0
            type_loss_sum = 0
            # 从2个epoch后每个epoch计算一次软标签
            #if epoch >=2:
             #   soft_labels_all =  self.generate_soft_label(model)
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                model.training = True
                model.train()
                idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
                
                bin_weights = self.gce_bin_weight[idx].to(self.device)
                type_weights = self.gce_type_weight[idx].to(self.device)
                
                # 软标签聚类
                if epoch >=3:
                    labels = soft_labels_all[idx].to(self.device)
                

                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights))
                if epoch>=2 and step%30 ==0:
                    soft_labels_all =  self.generate_soft_label(model)
                type_logits, bin_logits,labels,sb = model(input_ids, attention_mask, valid_pos,labels,flage=1)    
                labels = labels.long()
                bin_labels = labels.clone()
                bin_labels[labels > 0] = 1
                type_labels = labels - 1
                type_labels[type_labels < 0] = -100
                type_loss1 = loss_fct(type_logits.view(-1, self.num_labels-1), type_labels.view(-1), sb)
                type_loss = type_loss1.sum()
                
                if step %10 ==0:
                  y_pred, _ = self.eval(model, self.eval_dataloader)
                  #print(self.y_true)
                  #print(y_pred)
                  error_sample = []
                  for i1,j1 in zip(y_pred,self.y_true):
                    for index in range(len(i1)):
                      if i1[index].split('-')[-1]!=j1[index].split("-")[-1]:
                        error_sample.append((i1[index].split('-')[-1],j1[index].split('-')[-1]))
                  cnt = Counter(error_sample)
                  cnt = cnt.most_common()
                  for tup_c in cnt:
                    print(tup_c)
                  
                  
                  
                  print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                  prec,rec,f1 =  self.performance_report(self.y_true, y_pred)
                  if float(f1)>bestF1:
                    
                    bestF1 = float(f1)
                  print("bestF1",bestF1)
                  viz.line([float(prec)], [x_scale], win="prec",update="append")
                  viz.line([float(rec)], [x_scale], win="rec",update="append")
                  viz.line([float(f1)], [x_scale], win="f1",update="append")
                  x_scale+=1
                  print('lr:',scheduler.get_lr())
                  if float(f1)>max_f1:
                    # 保存模型
                    torch.save(model.state_dict(),os.path.join(self.temp_dir, f"Best_model{model_idx}.pt"))
                type_loss_sum += type_loss.item()
                bin_loss1 = loss_fct(bin_logits.view(-1, 1), bin_labels.view(-1), sb)
                bin_loss = bin_loss1.sum() 
                bin_loss_sum += bin_loss.item()
                loss1 = type_loss +bin_loss 
                loss1.backward()
                # 这里，有的影响吧
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                viz.line([float(loss1)], [i], win="loss",update="append")
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                # 软标签训练
                if False:
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights = tuple(t[:, :max_len] for t in \
                            (input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights))
                    
                    type_logits, bin_logits,labels,sb = model(input_ids, attention_mask, valid_pos,labels,flage=0)
                    
                    #labels = labels[valid_pos > 0]
                    #bin_weights = bin_weights[valid_pos > 0]
                    #type_weights = type_weights[valid_pos > 0]

                    bin_labels = labels.clone()
                    bin_labels[labels > 0] = 1
                    type_labels = labels - 1
                    type_labels[type_labels < 0] = -100
                    
                    

                    type_loss = loss_fct(type_logits.view(-1, self.num_labels-1), type_labels.view(-1), type_weights).sum()
                    
                    # print(type_loss)
                    type_loss_sum += type_loss.item()

                    bin_loss = loss_fct(bin_logits.view(-1, 1), bin_labels.view(-1), bin_weights).sum()
                    bin_loss_sum += bin_loss.item()
                    
                    loss = type_loss + bin_loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if (step+1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                
                    
                i += 1  



        pred_model =  GBRAINModel.from_pretrained(self.args.pretrained_model, num_labels=self.num_labels-1,hidden_dropout_prob=self.args.dropout, attention_probs_dropout_prob=self.args.dropout)
        pred_model.load_state_dict(torch.load(os.path.join(self.temp_dir, f"Best_model{model_idx}.pt")))

        eval_sampler = SequentialSampler(self.train_data)
        eval_dataloader = DataLoader(self.train_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        y_pred, pred_probs = self.eval(pred_model, eval_dataloader)
        torch.save({"pred_probs": pred_probs}, os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt"))



    # compute soft labels for self-training on entity type classes
    def soft_labels(self, model, entity_threshold=0.8):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.eval_batch_size)
        model.eval()

        type_preds = []
        indices = []
        for batch in tqdm(train_dataloader, desc="Computing soft labels"):
            idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
            type_distrib = torch.zeros(input_ids.size(0), self.max_seq_length, self.num_labels-1).to(self.device)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))
            with torch.no_grad():
                type_logits, bin_logits,_,_ = model(input_ids, attention_mask, valid_pos)
                type_pred = F.softmax(type_logits, dim=-1)
                entity_prob = torch.sigmoid(bin_logits)
                type_pred[entity_prob.squeeze() < entity_threshold] = 0
                type_distrib[:, :max_len][valid_pos > 0] = type_pred
                type_preds.append(type_distrib)

            indices.append(idx)
        
        type_preds = torch.cat(type_preds, dim=0)
        all_idx = torch.cat(indices)

        type_distribution = torch.zeros(len(self.train_data), self.max_seq_length, self.num_labels-1)
        for idx, type_pred in zip(all_idx, type_preds):
            type_distribution[idx] = type_pred

        type_distribution = type_distribution.view(-1, type_distribution.size(-1))
        valid_rows = type_distribution.sum(dim=-1) > 0
        weight = type_distribution[valid_rows]**2 / torch.sum(type_distribution[valid_rows], dim=0)
        target_distribution = (weight.t() / torch.sum(weight, dim=-1)).t()
        type_distribution[valid_rows] = target_distribution
        type_distribution = type_distribution.view(len(self.train_data), self.max_seq_length, self.num_labels-1)
        
        return type_distribution

    # self-training with augmentation
    
    # obtain model predictions on a given dataset
    def eval(self, model, eval_dataloader):
        model = model.to(self.device)
        model.eval()
        model.trainint= False
        y_pred = []
        pred_probs = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            _, input_ids, attention_mask, valid_pos, _ = tuple(t.to(self.device) for t in batch)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos))

            with torch.no_grad():
                logits, bin_logits,_,_ = model(input_ids, attention_mask, valid_pos)
                
                entity_prob = torch.sigmoid(bin_logits)
                type_prob = F.softmax(logits, dim=-1) * entity_prob
                non_type_prob = 1 - entity_prob
                type_prob = torch.cat([non_type_prob, type_prob], dim=-1)
                
                preds = torch.argmax(type_prob, dim=-1)
                preds = preds.cpu().numpy()
                pred_prob = type_prob.cpu()

            num_valid_tokens = valid_pos.sum(dim=-1)
            i = 0
            for j in range(len(num_valid_tokens)):
                pred_probs.append(pred_prob[i:i+num_valid_tokens[j]])
                y_pred.append([self.inv_label_map[pred] for pred in preds[i:i+num_valid_tokens[j]]])
                i += num_valid_tokens[j]
        

        return y_pred, pred_probs

    # print out ner performance given ground truth and model prediction
    def performance_report(self, y_true, y_pred):
        for i in range(len(y_true)):
            if len(y_true[i]) > len(y_pred[i]):
                print(f"Warning: Sequence {i} is truncated for eval! ({len(y_pred[i])}/{len(y_true[i])})")
                y_pred[i] = y_pred[i] + ['O'] * (len(y_true[i])-len(y_pred[i]))
        report = classification_report(y_true, y_pred, digits=3)


        # 获取micro f1
        rep = report.split('\n')
        f1 = rep[-4].split()[-2]
        rec = rep[-4].split()[-3]
        prec = rep[-4].split()[-4]

        # print(rep)
        # 获取weighted f1
        # rep = report.split()
        
        # f1 = rep[-2]
        # rec = rep[-3]
        # prec = rep[-4]
        
        print(report)
        
        return prec,rec,f1

    # save model, tokenizer, and configs to directory
    def save_model(self, model, model_name, save_dir):
        print(f"Saving {model_name} ...")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, model_name))
        self.tokenizer.save_pretrained(save_dir)
        model_config = {"max_seq_length": self.max_seq_length, 
                        "num_labels": self.num_labels, 
                        "label_map": self.label_map}
        json.dump(model_config, open(os.path.join(save_dir, "model_config.json"), "w"))

    # load model from directory
    def load_model(self, model_name, load_dir):
        self.model.load_state_dict(torch.load(os.path.join(load_dir, model_name)))
    # 生成 软标签
    def generate_soft_label(self, model, entity_threshold=0.8):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.eval_batch_size)
        model.eval()

        type_preds = []
        indices = []
        for batch in tqdm(train_dataloader, desc="Computing soft labels"):
            idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
            type_distrib = torch.zeros(input_ids.size(0), self.max_seq_length, 1,dtype=torch.long).to(self.device)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))
            with torch.no_grad():
                type_logits, bin_logits,_,_ = model(input_ids, attention_mask, valid_pos)
                entity_prob = torch.sigmoid(bin_logits)
                type_prob = F.softmax(type_logits, dim=-1) * entity_prob
                non_type_prob = 1 - entity_prob
                type_prob = torch.cat([non_type_prob, type_prob], dim=-1)
                
                # 概率小于 0.9 的  标签设置为-100
                value_and_index = torch.max(type_prob,dim=-1)
                value = value_and_index[0]
                preds = value_and_index[1]


                # print(value)
                # print(preds)
                preds[value<0.9] = -100
               
               # preds[preds==-100] = labels[valid_pos>0][preds==-100]
                
                
                # preds = torch.argmax(type_prob, dim=-1)
                soft_labels = preds.unsqueeze(1)
                

                type_distrib[:, :max_len][valid_pos > 0] = soft_labels
                

                type_preds.append(type_distrib)


            indices.append(idx)
        
        type_preds = torch.cat(type_preds, dim=0)
        all_idx = torch.cat(indices)

        type_distribution = torch.zeros(len(self.train_data), self.max_seq_length, 1,dtype=torch.long)
        for idx, type_pred in zip(all_idx, type_preds):
            type_distribution[idx] = type_pred 
        return type_distribution