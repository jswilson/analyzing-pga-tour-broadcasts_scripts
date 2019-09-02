# this script is based on pytorch documentation located here:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# 
# I've made use of this setup on image classification tasks with a pretrained
# model frequently, and it definitely was useful again for this analysis

import sys

# the first (and only) argument passed in is the folder containing the
# training and validation images.  We assume a folder structure of:
# root/
# └── train/
#     └── comm/
#     └── regular-broadcast/
#     └── pt/
#     └── segment/
# └── valid/
#     └── comm/
#     └── regular-broadcast/
#     └── pt/
#     └── segment/
data_dir = sys.argv[1]

num_classes = 4
batch_size = 32

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, filenames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_resnet(num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def get_params_to_update(model_ft, lr=0.001):
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    optimizer_ft = optim.Adam(params_to_update, lr=lr)
    return optimizer_ft


# init the model
model_ft, input_size = initialize_resnet(num_classes, feature_extract, use_pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# init data transforms (very importantly, normalize to imagenet)
# we also add in some various transforms to try and make the model more
# resilient
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(10.0, scale=(1, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# init datasets and dataloaders
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'valid']}

# Send the model to GPU
model_ft = model_ft.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# begin with 3 epochs with the default learning rate of 0.001
# we're starting with *only* training the final layer of the model
optimizer_ft = get_params_to_update(model_ft)
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=3)

# do more 3 epochs, but lower the learning rate to 0.0003
optimizer_ft = get_params_to_update(model_ft, lr=0.0003)
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=3)

# "unfreeze" the model
for param in model_ft.parameters():
    param.requires_grad = True
set_parameter_requires_grad(model_ft, False)
feature_extract = False

# then run 3 more epochs where we finetune the *entire* model, not just the
# last layer
optimizer_ft = get_params_to_update(model_ft, lr=0.0003)
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=3)