from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from collections import OrderedDict

def download_mnist(save_path):
    torchvision.datasets.MNIST(root=save_path,train=True,download=True)
    torchvision.datasets.MNIST(root=save_path,train=False,download=True)
    return save_path

def load_mnist(batch_size=64,path='',img_size=32):
    if img_size != 32:
        transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()]
        )
    else:
        transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=path,train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root=path,train=False,download=True,transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    dataloaders = {"train":trainloader,"val":testloader}
    dataset_sizes = {"train":60000,"val":10000}
    return dataloaders,dataset_sizes

def download_cifar10(save_path):
    torchvision.datasets.CIFAR10(root=save_path,train=True,download=True)
    torchvision.datasets.CIFAR10(root=save_path,train=False,download=True)
    return save_path

def load_cifar10(batch_size=64,pth_path='./data',img_size=32):
    if img_size!=32:
        transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((img_size,img_size))
            ,transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Pad(padding = 4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=pth_path, train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=pth_path, train=False,download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    dataloaders = {"train":trainloader,"val":testloader}
    dataset_sizes = {"train":60000,"val":10000}
    return dataloaders,dataset_sizes

def download_cifar100(save_path):
    torchvision.datasets.CIFAR100(root=save_path,train=True,download=True)
    torchvision.datasets.CIFAR100(root=save_path,train=False,download=False)
    return save_path

def load_cifar100(batch_size,pth_path,img_size):
    if img_size!=32:
        transform = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((img_size,img_size))
            ,transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Pad(padding = 4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root=pth_path,train=True,download=False,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    testset = torchvision.datasets.CIFAR100(root=pth_path,train=False,download=False,transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)
    dataloaders = {"train":trainloader,"val":testloader}
    dataset_size ={"train":50000,"val":10000}
    return dataloaders,dataset_size
def test_model(model,dataloaders,dataset_sizes,criterion):
    print("validation model:")
    phase = "val"
    model.cuda()
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for inputs,labels in tqdm(dataloaders[phase]):
            inputs,labels = inputs.cuda(),labels.cuda()
            outputs = model(inputs)
            _,preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(preds == labels.data)
        epoch_loss = running_loss/dataset_sizes[phase]
        epoch_acc = running_acc / dataset_sizes[phase]
        epoch_acc = epoch_acc.item()
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
    return epoch_acc,epoch_loss

def WriteData(savePath, msg):

    full_path = savePath + '/Accuracy.txt' 
    file = open(full_path, 'a')
    file.write(msg)   
    # file.close()

def train_model_jiang(model, dataloaders, dataset_sizes,ratio, type,quantize,pattern,criterion, optimizer, name,scheduler=None, num_epochs=100,rerun=False):
    if rerun == True:
        print(num_epochs)
        since = time.time()
        model = torch.load('../../hdd/hdd_o/pth/AlexNet/ratio=0.8/Activation/test_17.pth')
        if type == 'activation':
            savePth = '../../hdd/hdd_o/pth/'+name+'/ratio='+str(ratio)+'/Activation'
        else:
            if pattern == 'retrain':
                savePth = '../../hdd/hdd_o/pth/'+name+'/ratio='+str(ratio)+'/ActivationWeight'
            elif pattern == 'train':
                savePth = '../../hdd/hdd_o/pth/' + name + '/ratio=0' + '/Weight'
            elif pattern == 'DPTrain':
                savePth = '../../hdd/hdd_o/pth/' + name + '/ratio=0' + '/DynamicPrune'
        model.cuda()
        WriteData(savePth,'ratio='+str(ratio)+'\n'+'allEpoch=' + str(num_epochs)+'\n')
        best_model_wts = copy.deepcopy(model)
        best_acc = 0.0
        weight_deacy = 0.01
        model.cuda()
        for epoch in range(23,num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            print('the %d lr:%f'%(epoch+1,optimizer.param_groups[0]['lr']))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    print('val stage')
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                i = 0
                # loss_a = 0
                # p = 0
                for data in dataloaders[phase]:
                    inputs,labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss_re = 0.0  
                        for name,par in model.named_parameters():
                            loss_re = loss_re + weight_deacy * 0.5* torch.sum(torch.pow(par,2))
                        loss = loss + loss_re
                        loss_a = loss.item()
                        print('[%d ,%5d] loss:%.3f'%(epoch+1,i+1,loss_a))
                        # loss_a = 0
                        i += 1
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # path = savePth + '/testtt_{}.pth'.format(epoch + 1)
                    # torch.save(model, path)
                    # break

                if phase == 'train' and scheduler is not None:
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # epoch_loss = running_loss / p
                # epoch_acc = running_corrects.double() / p
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' :
                    path = savePth + '/test_{}.pth'.format(epoch + 1)
                    torch.save(model, path)
                    WriteData(savePth,str((round(float(epoch_acc), 4)) * 100) + '%-' + 'epoch=' + str(epoch + 1) + '\n')
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        path = savePth + '/best.pth'
        torch.save(best_model_wts, path)
        
    if rerun == False:
        since = time.time()
        # best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        if type == 'activation' and quantize == True and ratio == 0:
            savePth = './pth/'+name+'/ratio='+str(ratio)+'/Quantize'
        else:
            savePth = './pth/' + name + '/ratio=' + str(ratio) + '/Activation'
        '''
        else:
            if pattern == 'retrain':
                savePth = './pth/'+name+'/ratio='+str(ratio)+'/ActivationWeight'
            elif pattern == 'train':
                savePth = './pth/' + name + '/ratio=0' + '/Weight'
            elif pattern == 'DPTrain':
                savePth = './pth/' + name + '/ratio=0' + '/DynamicPrune'
        '''
        model.cuda()
        WriteData(savePth,'ratio='+str(ratio)+'\n'+'allEpoch=' + str(num_epochs)+'\n')
        weight_deacy = 0.001
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            print('the %d lr:%f'%(epoch+1,optimizer.param_groups[0]['lr']))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    print('val stage')
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                i = 0
                # loss_a = 0
                # p = 0
                for data in dataloaders[phase]:
                    inputs,labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss_re = 0.0  
                        for name, par in model.named_parameters():
                            loss_re = loss_re + weight_deacy * 0.5 * torch.sum(torch.pow(par, 2))
                        loss = loss + loss_re
                        loss_a = loss.item()

                        print('[%d ,%5d] loss:%.3f'%(epoch+1,i+1,loss_a))
                        # loss_a = 0
                        i += 1
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # path = savePth + '/testtt_{}.pth'.format(epoch + 1)
                    # torch.save(model, path)
                    # break

                if phase == 'train' and scheduler is not None:
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # epoch_loss = running_loss / p
                # epoch_acc = running_corrects.double() / p
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' :
                    if epoch_acc > best_acc:
                        best_epoch = epoch
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model)
                        path = savePth + '/test_{}.pth'.format(epoch + 1)
                        torch.save(model, path)
                        WriteData(savePth,str((round(float(epoch_acc), 4)) * 100) + '%-' + 'epoch=' + str(epoch + 1) + '\n')
                    # if epoch_acc > best_acc:
                    #     best_acc = epoch_acc
                    #     best_model_wts = copy.deepcopy(model)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        path = savePth + '/best.pth'
        WriteData(savePth, str((round(float(best_acc), 4)) * 100) + '%-'  + str(best_epoch + 1) + '\n')
        torch.save(best_model_wts, path)
    return model











