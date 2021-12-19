import torch

from typing import Any


def validate(model: Any, test_loader: Any, device: Any, criterion: Any, summarywriter: Any) -> None:
    """
    Test network, calculate loss

    Parameters
    ----------
    model: Any
        trained model
    test_loader: Any
        DataLoader from test dataset images
    device: Any
        type of device network was trained on (cuda or CPU)
    criterion: Any
        loss calculation type
    summarywriter: Any
        TensorBoard summarywriter
    Returns
    -------

    """
    correct = 0
    total = 0
    running_loss = 0.0
    iteration = 0
    classes = test_loader.dataset.classes
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            summarywriter.add_scalar("Validation Loss", running_loss, iteration)
            iteration += 1
        print(
            "Accuracy of the network on the 40 test images: %d %%"
            % (100 * correct / total)
        )
    summarywriter.close()

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))