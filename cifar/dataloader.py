class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(logits, label):
    pred = logits.argmax(dim=1)
    ret = torch.eq(pred, label).sum().float() / pred.shape[0]
    return ret


class DataReader(Dataset):
    
    def __init__(self, input_path, size=150, is_train=False):
        self.size = size
        self.classes = ['AP2', 'AP3', 'AP4', 'AP5', 'PLAX', 'PSAX-AP', 'PSAX-AV', 'PSAX-MID',
                 'PSAX-MV', 'NONE']
        self.input_path = input_path
        
        self.reverse_label = dict()
        for i, x in enumerate(self.classes):
            self.reverse_label[x] = i
        
        self.sampes = []
        for x in self.classes:
            files = os.listdir(self.input_path + "/" + x)
            for file in files:
                self.sampes.append((self.input_path + "/" + x + "/" + file,
                                    self.reverse_label[x]))
        np.random.shuffle(self.sampes)
        if is_train:
            self.transform =  transforms.Compose([
                                        transforms.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0)),
                                        transforms.RandomRotation(degrees=20, fill=0),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor()
                                    ])
        else:
            self.transform =  transforms.Compose([
                                        transforms.Resize(size=(size, size)),
                                        transforms.ToTensor()
                                    ])
            
    def __len__(self):
        return len(self.sampes)
    
    def __getitem__(self, idx):
        file, label = self.sampes[idx]
        image = Image.open(file)
        ret = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.int64)
        return ret, label

class ImageModel(nn.Module):
    
    def __init__(self, model_type):
        super(ImageModel, self).__init__()
        
        middle_size = 2048 
        base_model = None
        if model_type == "xception":
            base_model = pretrainedmodels.xception(1000, pretrained="imagenet")
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            middle_size = 2048
        elif model_type == "vgg":
            base_model = pretrainedmodels.vgg16(num_classes=1000, pretrained='imagenet')
            self.base_model = nn.Sequential(*list(base_model.children())[1:2])
            middle_size = 512
        elif model_type == "inception":
            base_model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
            self.base_model = nn.Sequential(*list(base_model.children())[:-2])
            middle_size = 1536
        
        self.middle_model = nn.Sequential(
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Dropout(),
                                nn.Linear(middle_size, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout()
                            )
        
        self.top_model =  nn.Linear(512, 10)
        
    def forward(self, inputs):
        x = self.base_model(inputs)
        x = self.middle_model(x)
        x = self.top_model(x)
        
        return x 

def train_one_epoch(epoch, epochs, model, train_loader, device, criterion, optimizer, log_fp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracys = AverageMeter()
    model.train()    
    end = time.time()

    progress = tqdm(train_loader, desc='Epoch:[{0}/{1}][training stage]'.format(epoch + 1, epochs))
    for data, label in progress:
        data_time.update(time.time() - end)
        data = data.to(device)
        label = label.to(device)
    
        logits = model(data)
        loss = criterion(logits, label)
        losses.update(loss.item())
        accuracys.update(accuracy(logits.data, label.data).item(), data.data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_fp.write(str(losses.avg) + "\t" + str(accuracys.avg) + "\n")
        progress.set_description(
                   'Epoch:[{0}/{1}][training stage]'.format(epoch + 1, epochs) + ' Batch Time:' + 
                   str(round(batch_time.avg, 1)) + ' Data Time:' + 
                   str(round(data_time.avg, 1)) + ' Loss:' + 
                   str(round(losses.avg, 2)) + ' accu:' + 
                   str(round(accuracys.avg, 2)) +  ' lr:' + 
                   str(optimizer.param_groups[0]['lr'])
                )
        
    progress.close()
    return losses.avg, accuracys.avg

    
@torch.no_grad()   
def valid_one_epoch(epoch, epochs, model, valid_loader, device, criterion, optimizer, log_fp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracys = AverageMeter()
    model.eval()
    end = time.time()

    progress = tqdm(valid_loader, desc='Epoch:[{0}/{1}][validate stage]'.format(epoch + 1, epochs))
    for data, label in progress:
        data_time.update(time.time() - end)
        data = data.to(device)
        label = label.to(device)

        logits = model(data)
        loss = criterion(logits, label)
        # item是取tensor的值, 针对scalar
        losses.update(loss.item())
         # .data返回副本, 不记录梯度, data.data.shape[x]是scalar
        accuracys.update(accuracy(logits.data, label.data).item(), data.data.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()
        log_fp.write(str(losses.avg) + "\t" + str(accuracys.avg) + "\n")
        progress.set_description(
                   'Epoch:[{0}/{1}][validate stage]'.format(epoch + 1, epochs) + ' Batch Time:' + 
                   str(round(batch_time.avg, 1)) + ' Data Time:' + 
                   str(round(data_time.avg, 1)) + ' Loss:' + 
                   str(round(losses.avg, 2)) + ' accu:' + 
                   str(round(accuracys.avg, 2))
                )

    progress.close()
    return losses.avg, accuracys.avg
    
    
def train(train_path, valid_path, test_path, model_path, log_path, lr, batch_size, epochs, gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if os.path.exists(model_path):
        shutil.rmtree(model_path)    
    os.makedirs(model_path)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    train_log = log_path + "/train_log.txt"
    valid_log = log_path + "/valid_log.txt"
    test_log = log_path + "/test_log.txt"
    train_log_fp = open(train_log, "w")
    valid_log_fp = open(valid_log, "w")
    test_log_fp = open(test_log, "w")

    train_loader = DataLoader(DataReader(train_path, size=299, is_train=True), batch_size=batch_size, shuffle=True, 
                              num_workers=32, drop_last=True)
    valid_loader = DataLoader(DataReader(valid_path,  size=299), batch_size=batch_size, shuffle=True, 
                              num_workers=32, drop_last=True)
    test_loader = DataLoader(DataReader(test_path,  size=299), batch_size=batch_size, shuffle=True, 
                              num_workers=32, drop_last=True)
    
    device = torch.device("cuda:" + str(gpu_index))
    model = ImageModel(model_type="inception")
    model = nn.DataParallel(model)
    model.to(device=device)
    
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(epoch, epochs, model, train_loader, device, 
                                                criterion, optimizer, train_log_fp)
        val_loss, val_acc = valid_one_epoch(epoch, epochs, model, valid_loader, device, 
                                            criterion, optimizer, valid_log_fp)
        valid_one_epoch(epoch, epochs, model, test_loader, device, criterion, optimizer, test_log_fp)
        scheduler.step(val_loss)
        state = {
                 'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer':  optimizer.state_dict(),
                 'train_loss': train_loss,
                 'train_acc': train_acc,
                 'val_loss': val_loss,
                 'val_acc': val_acc
                }
        torch.save(state, model_path + "/" + "xception_" + str(epoch) + "_epoch_valid_acc_" + 
                   str(val_acc)+".pth")
        
    train_log_fp.close()
    test_log_fp.close()
    test_log_fp.close()