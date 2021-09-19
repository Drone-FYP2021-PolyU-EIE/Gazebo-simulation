#-------------------------------------#
#       train the dataset
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo4_tiny import YoloBody
from yolo_training import LossHistory, YOLOLoss, weights_init
from dataloader import YoloDataset, yolo_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
# Get class and a priori box
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])

def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc='Epoch {}/{}'.format(epoch + 1,Epoch),postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            #----------------------#
            #   Clear gradient
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward propagation
            #----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   Calculate the loss
            #----------------------#
            for i in range(2):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            #----------------------#
            #   Backward propagation
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val,desc='Epoch {}/{}'.format(epoch + 1,Epoch),postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(2):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

if __name__ == "__main__":
    #-------------------------------#
    #   Type of attention mechanism used
    #   phi = 0 not using attention
    #   phi = 1 is SE
    #   phi = 2 is CBAM
    #   phi = 3 is ECA
    #-------------------------------#
    phi = 0
    #-------------------------------#
    #    Use Cuda or not
    #    if not set it to False
    #-------------------------------#
    Cuda = True
    #------------------------------------------------------#
    # Whether to normalize the loss to change the size of the loss
    # Used to determine whether to calculate the final loss by dividing batch_size or dividing by the number of positive samples
    #------------------------------------------------------#
    normalize = False
    #-------------------------------#
    #    input shape size
    #   low memory please use 416x416 
    #   large memory please use 608x608
    #-------------------------------#
    input_shape = (416,416)
    #----------------------------------------------------#
    #    path for classes and anchor
    #    Before training, please modify the classes_path
    #----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'   
    #------------------------------------------------------#
    #    tricks application for  Yolov4
    #   mosaic  (True or False)
    #   Cosine_scheduler  (True or False)
    #   label_smoothing (normally small thatn 0.01)
    #------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    smoooth_label = 0

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)

    #------------------------------------------------------#
    #    build the yolo model
    #    modify the classes_path and the corresponding txt file before training
    #------------------------------------------------------#
    model = YoloBody(len(anchors[0]), num_classes, phi)
    weights_init(model)

    model_path = "/home/lai/Desktop/custom_deep_learning/yolov4-tiny/model_data/yolov4_tiny_weights_coco.pth"
    # Speed up the efficiency of model training
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize)
    loss_history = LossHistory("logs/")

    #----------------------------------------------------#
    #   Get image path and label
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    # The division of the verification set is carried out in the train.py code
    # It is normal that there is no content in  2007_test.txt and 2007_val.txt. Training will not be used.
    # In the current division method, the ratio of the validation set to the training set is 1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    # The main feature extraction network feature is general, freezing training can speed up the training speed
    # It can also prevent the weight from being destroyed in the early stage of training.
    # Init_Epoch is the initial generation
    # Freeze_Epoch is the generation of freeze training
    # Epoch Total Training Generation
    # Prompt OOM or insufficient video memory, please reduce Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-3
        Batch_size      = 32
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")
        #------------------------------------#
        #   Freeze a certain part of training
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        lr              = 1e-4
        Batch_size      = 16
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100

        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")
        #------------------------------------#
        #   Train after unfreeze
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
