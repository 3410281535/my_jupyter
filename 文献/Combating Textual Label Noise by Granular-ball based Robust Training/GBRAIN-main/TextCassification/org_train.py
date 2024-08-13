import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings
warnings.filterwarnings("ignore")

from config import opt
import random

#from visdom import Visdom
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import pickle
from loss_2 import *
from MyDataset import *
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from torchnlp.samplers import BucketBatchSampler

from model import Model
from utils import load_data, MarginLoss, collate_fn, FocalLoss ,loadGloveModel,create_embeddings_matrix


print('gpu数量:',torch.cuda.device_count())

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(best_acc, model, optimizer, epoch):
    print('Best Model Saving...')
    model_state_dict = model.state_dict()
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save({
        'model_state_dict': model_state_dict,  # 网络参数
        'global_epoch': epoch,  # 最优对应epoch
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
        'best_acc': best_acc,  # 最好准确率
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))

    """
    相关参数
    """
def getitem_by_index(dataset, index):
    """
    根据数据集原始index获取经验池数据
    Args:
        dataset:
        index:

    Returns:

    """
    imgs = []
    targets = []
    original_targets = []
    for i,j in index:

        img, target, original_target = np.array(dataset.datafram["text"][i]), dataset.datafram["noise_label"][i], dataset.datafram["label"][i]


        # img = Image.fromarray(img)

        # if dataset.transform is not None:
        #     img = dataset.transform(img)
        #
        # if dataset.target_transform is not None:
        #     target = dataset.target_transform(target)
        #     original_target = dataset.target_transform(original_target)
        imgs.append(img)
        targets.append(j)
        original_targets.append(original_target)

    # imgs = torch.from_numpy(np.array(imgs, dtype=np.float32))
    # targets = torch.from_numpy(np.array(targets, dtype=np.float32))
    # original_targets = torch.from_numpy(np.array(original_targets, dtype=np.float32))

    return imgs, targets, original_targets, index
    
def l2_regularization(model, l2_alpha):
    p_list=[]
    for name in model.state_dict():
        if name !="embedding.weight" and ("bias" not in name):
            
            p = model.state_dict()[name]
            p_list.append((p**2).sum())
        
        
    return l2_alpha * sum(p_list)*1/2

    


if __name__ == '__main__':
    # 固定随机种子
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    DATA_TYPE, FINE_GRAINED, TEXT_LENGTH = opt.data_type, opt.fine_grained, opt.text_length
    ROUTING_TYPE, LOSS_TYPE, EMBEDDING_TYPE = opt.routing_type, opt.loss_type, opt.embedding_type
    CLASSIFIER_TYPE, EMBEDDING_SIZE, NUM_CODEBOOK = opt.classifier_type, opt.embedding_size, opt.num_codebook
    NUM_CODEWORD, HIDDEN_SIZE, IN_LENGTH = opt.num_codeword, opt.hidden_size, opt.in_length
    OUT_LENGTH, NUM_ITERATIONS, DROP_OUT, BATCH_SIZE = opt.out_length, opt.num_iterations, opt.drop_out, opt.batch_size
    NUM_REPEAT, NUM_EPOCHS, NUM_STEPS, PRE_MODEL = opt.num_repeat, opt.num_epochs, opt.num_steps, opt.pre_model

    # prepare dataset
    sentence_encoder, label_encoder, train_dataset, test_dataset = load_data(DATA_TYPE, preprocessing=True,
                                                                             fine_grained=FINE_GRAINED, verbose=True,
                                                                text_length=TEXT_LENGTH)
    
    # print(sentence_encoder.vocab)
    # print(sentence_encoder.encode('currency').item())
    # exit()


    embeddings_matrix = None
    if os.path.exists('./{}_matrix.pkl'.format(DATA_TYPE)):
        print('./{}_matrix.pkl'.format(DATA_TYPE),'已存在')
        with open('./{}_matrix.pkl'.format(DATA_TYPE),'rb') as f:
            embeddings_matrix = pickle.load(f)
    else:
        glove_model = loadGloveModel("glove.6B.300d.txt")
        embeddings_matrix = create_embeddings_matrix(glove_model, sentence_encoder, 300)        
        with open('./{}_matrix.pkl'.format(DATA_TYPE),'wb') as f:
            pickle.dump(embeddings_matrix,f)


    

    train_dataset = MyDataset(train_dataset,opt.noise_p) # 这里的数据已经是加噪后的数据集，噪声比例opt.noise_p 这时是0.1
    test_dataset = MyDataset(test_dataset,0)
    VOCAB_SIZE, NUM_CLASS = sentence_encoder.vocab_size, label_encoder.vocab_size
    print("纯度：",opt.purity)
    print("[!] vocab_size: {}, num_class: {}".format(VOCAB_SIZE, NUM_CLASS))
    train_iterator = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_iterator = DataLoader(test_dataset,BATCH_SIZE*2)

    # train_sampler = BucketBatchSampler(train_dataset, BATCH_SIZE, False, sort_key=lambda row: len(row['text']))
    # train_iterator = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    # test_sampler = BucketBatchSampler(test_dataset, BATCH_SIZE * 2, False, sort_key=lambda row: len(row['text']))
    # test_iterator = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, IN_LENGTH, OUT_LENGTH, NUM_CLASS, ROUTING_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE, DROP_OUT,embeddings_matrix)

    if PRE_MODEL is not None:
        model_weight = torch.load('epochs/{}'.format(PRE_MODEL), map_location='cpu')
        model_weight.pop('classifier.weight')
        model.load_state_dict(model_weight, strict=False)
    if LOSS_TYPE == 'margin`':
        loss_criterion = [MarginLoss(NUM_CLASS)]
    elif LOSS_TYPE == 'focal':
        loss_criterion = [FocalLoss()]
    elif LOSS_TYPE == 'cross':
        loss_criterion = [CrossEntropyLoss()]
    elif LOSS_TYPE == 'mf':
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss()]
    elif LOSS_TYPE == 'mc':
        loss_criterion = [MarginLoss(NUM_CLASS), CrossEntropyLoss()]
    elif LOSS_TYPE == 'fc':
        loss_criterion = [FocalLoss(), CrossEntropyLoss()]
    elif LOSS_TYPE == 'GCE':
        loss_criterion = [GCELoss(0.9)]
    else:
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss(), CrossEntropyLoss()]
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to(device), True

    if PRE_MODEL is None:
        optim_configs = [{'params': model.embedding.parameters(), 'lr': 1e-4},
                         {'params': model.features.parameters(), 'lr': 1e-3},
                         #{'params': model.fc1.parameters(), 'lr': 1e-3 },
                         {'params': model.classifier.parameters(), 'lr': 1e-3}]
    else:
        for param in model.embedding.parameters():
            param.requires_grad = False
        for param in model.features.parameters():
            param.requires_grad = False
        optim_configs = [{'params': model.classifier.parameters(), 'lr': 1e-4}]
    optimizer = Adam(optim_configs, lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS*0.9), int(NUM_EPOCHS*0.9)+2], gamma=0.5)

    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    # record statistics
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    # record current best test accuracy
    best_acc = 0
   

    # config the visdom figures
    if FINE_GRAINED and DATA_TYPE in ['reuters', 'yelp', 'amazon']:
        env_name = DATA_TYPE + '_fine_grained'
    else:
        env_name = DATA_TYPE
    current_step = 1
    print("batch_size",opt.batch_size)
    purity_d = opt.purity 
    for epoch in range(1, NUM_EPOCHS + 1):
        #if epoch > 30 and epoch < 35:
         #   opt.drop_ball +=0.1

       
        
        model.train()
		# 定义经验池
        exp_pool = []
        training_loss = 0.0
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        print("--" * 20)
          # [[2,3],[4,9],[1,5],[6,8]]
        for i, train_data in enumerate(train_iterator):
            #print("- -- -- -- -- -- -" * 2)
            current_step+=1
			# 经验池里面取出的数据存放的列表
            exp_data, exp_target, exp_original_target, exp_index = [], [], [], []
            x, (labels, index), original_labels = train_data
            x, labels, index, original_labels = x.to(device), labels.to(device), index.to(device), original_labels.to(device)

            y, target, exp = model(x, labels, index, original_labels, 1, purity_d)
            loss = loss_criterion[0](y, target.long())#+l2_regularization(model, 2*1e-4)  
            training_loss += loss.item()  # 一个epoch的loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if current_step % 20 == 0:  # 没20次做一次评估
                model.eval()  # 进入测试，测试时不启用 Batch Normalization 和 Dropout
                num_correct = 0.
                test_total = 0
                with torch.no_grad():
                    for j, test_data in enumerate(test_iterator):
                        
                        test_img, (noise_label, idx), test_label=test_data
                        test_img, test_label = test_img.to(device), test_label.to(device)
						# 评估过程，不过粒球层，标志参数设置：
                        y, _, _ = model(test_img, test_label, [], [], 1, 0)
                        _, pred = torch.max(y.data, 1)  
                        test_correct = (pred == test_label).sum().item()
                        num_correct += test_correct
                        test_total += test_label.size(0)

                test_accuracy = num_correct / test_total * 100.
                print('[%d, %5d] train_loss: %.4f test_accuracy: %.2f test_correct:%d test_total:%d'
                      % (epoch, i + 1, training_loss / 20, test_accuracy, num_correct, test_total))
				# 将当前的评估信息写入文件
                with open("recluster/training_loss_acc"+str(opt.drop_ball)+".txt","a",encoding="utf-8") as f:
                        f.write('[%d, %5d] train_loss: %.4f test_accuracy: %.2f test_correct:%d test_total:%d'
                      % (epoch, i + 1, training_loss / 20, test_accuracy, num_correct, test_total)+"\n")
                


                # writer.add_scalar("Test/Accuracy", test_accuracy, i)

                if test_accuracy > best_acc:
                    
                    best_acc = test_accuracy
                    best_acc_loc = epoch
                    save_checkpoint(best_acc, model, optimizer, epoch)
                training_loss = 0.0

                model.train()  # 返回训练
                print('Current best acc:{:.2f}% at epoch{}'.format(best_acc, best_acc_loc))
                print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
				# 最好的评估信息写入文件
                with open("recluster/base_acc"+str(opt.drop_ball)+".txt","a",encoding="utf-8") as f:
                        f.write('Current best acc:{:.2f}% at epoch{}'.format(best_acc, best_acc_loc)+"\n")

                with open("recluster/base_acc_epoch"+str(opt.round)+".txt","w",encoding="utf-8") as f:
                        f.write('Current best acc:{:.2f}% at epoch{}'.format(best_acc, best_acc_loc)+"\n")
        lr_scheduler.step(epoch)
    print("完成")
