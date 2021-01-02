import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import sys
import argparse
import pathlib

def train(data_dir, save_path, arch, learn_rate = 0.001, hidden_units = 500, epochs = 3, gpu = True, verbose = True):
    if verbose:
        print(f'Data directory: {data_dir}')
        print(f'Checkpoint path: {save_path}')
        print(f'Architecture: {arch}')
        print(f'Learn rate: {learn_rate}')
        print(f'Number of hidden units: {hidden_units}')
        print(f'Number of epochs: {epochs}')
        print(f'Use GPU: {gpu}')
        
    # Define expected data structure
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Build network
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    print(f'Device used: {device}', flush=True)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_units = 25088
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        input_units = 25088
    else:
        raise Exception('Unknown basemodel type: ' + arch)

    for param in model.parameters():
        param.requires_grad = False

    classifier_definition = OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ])
    classifier = nn.Sequential(classifier_definition)
        
    model.classifier = classifier
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    #Train network
    print_every = 10

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}', flush=True)
        running_loss = 0
        test_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if ii % print_every == 0:
                test_loss = 0
                model.eval()
                with torch.no_grad():
                    equalsTotal = torch.empty(0, dtype=torch.uint8).to(device)
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        equalsTotal = torch.cat((equals, equalsTotal), 0)
                        
                accuracy = torch.mean(equalsTotal.type(torch.FloatTensor))
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(testloader):.3f}.. "
                      f"Validation accuracy: {accuracy.item()*100:.1f}%", flush=True)
                running_loss = 0
                model.train()

    # Do validation on the test set
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        equalsTotal = torch.empty(0, dtype=torch.uint8).to(device)
        for ii, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            equalsTotal = torch.cat((equals, equalsTotal), 0)

        test_accuracy = torch.mean(equalsTotal.type(torch.FloatTensor))
        print(f'Test Accuracy: {test_accuracy.item()*100:.1f}%', flush=True)
        print(f'Test Loss: {test_loss}', flush=True)

    model.train()

    checkpoint = {'basemodel': arch,
                  'basemodel_parameters_frozen': True,
                  'classifier_definition': classifier_definition,
                  'classifier': classifier.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'training_epoch_count': epochs,
                  'class_to_index': train_data.class_to_idx}

    torch.save(checkpoint, save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network on a set of images.')
    parser.add_argument('data_dir', type=pathlib.Path,
                        help='Path to data folder. Expected data folder contains 3 sub-directories (train, test, valid) each containing 102 subfolders numbered [1-102], storing image files.')
    parser.add_argument('--save_dir', type=pathlib.Path,
                        default='checkpoint.pth',
                        help='Path to save the definition of the saved neural network')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Base architecture of the neural network')
    parser.add_argument('--learn_rate', type=float, default=0.001,
                        help='Learn rate of the training.')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs in training.')
    parser.add_argument('--gpu', action='store_true',
                        help='Should use GPU / CUDA? (If available.)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose')

    args = parser.parse_args()
    
    train(data_dir = str(args.data_dir), save_path = str(args.save_dir), arch = args.arch, learn_rate = args.learn_rate, hidden_units = args.hidden_units, epochs = args.epochs, gpu = args.gpu, verbose = args.verbose)