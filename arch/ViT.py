

    def 
    

    def fit(self, x_train, y_train, x_test, y_test):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)
        testloader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), batch_size=16,
                                            shuffle=True, num_workers=1)
        for epoch in range(20):
            clf.train()
            trainloader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=16,
                                            shuffle=True, num_workers=1)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = clf(inputs.float())
                loss = criterion(outputs, labels.flatten().Long())
                loss.backward()
                optimizer.step()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}, acc: {((outpus >= 0.5).long().numpy() == labels.long()).sum().item() * 100. / labels.shape[0]}')
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = clf(images)
                    # the class with the highest energy is what we choose as prediction
                    predicted = (outputs.data > 0.5).long()
                    total += labels.size(0)
                    correct += (predicted == labels.long()).sum().item()
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')