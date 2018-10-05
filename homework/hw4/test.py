import torch

def test(
    net, 
    criterion, 
    test_data_loader, 
    device,
    debug=False
    ):
    """Testing setup for a single epoch
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        test_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        debug(bool): whether to use a debug mode
    """

    # Set to test mode.
    net.eval()

    # To monitor the testing process.
    running_loss = 0
    total_correct = 0
    total_samples = 0

    # Testing step.
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_data_loader):
            # Only test on a smaller subset if debug mode is true
            if debug and total_samples >= 10001:
                return
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels.long())
            
            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                    (batch_index + 1, curt_loss, accuracy))
    
    print("Testing [finished] finial accuracy: %.5f" % accuracy)