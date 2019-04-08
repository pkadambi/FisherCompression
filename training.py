from data import get_dataset
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import *
from utils.dataset_utils import *
cudnn.benchmark = True

torch.cuda.set_device(0)

c = LenetFashionMNISTConfig(binary = False,
                            n_epochs = 15)

model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
model = model(**model_config)
model.apply(xavier_initialize_weights)
model.apply(normal_initialize_biases)
model.cuda()

optimizer = optim.Adam(model.parameters())
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

transforms = { 'train': get_transform(name = c.dataset, augment=False),

               'test': get_transform(name = c.dataset, augment=False)}

train_data = get_dataset(c.dataset, 'train', transform = transforms['train'])


train_data, valid_data = generate_validation_split(train_data, validation_split=.05)

train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


test_data= get_dataset(c.dataset, 'test', transform = transforms['test'])
test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


lossval = np.array([])


model.train()
for epoch in range(c.n_epochs):
    # if i>0:
    #     break

    accuracy_ = 0
    for iter, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        # inputs = inputs.view(-1,28*28)

        target = target.cuda()

        output = model(inputs)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossval_ = loss.cpu().data.numpy()

        accuracy_ = accuracy(output, target)[0]

        if iter % c.print_interval ==0:
            print('Epoch %d | Train Loss %.3f | Acc %.2f' % (epoch+1, lossval_, accuracy_))

        lossval = np.hstack([lossval, lossval_])

    val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False)

    print('\nEpoch %d | Valid Loss %.3f | Valid Acc %.4f ' % (epoch + 1, val_loss, val_acc))
    print('Epoch %d | Test Loss %.3f | Test Acc %.2f \n' % (epoch + 1, test_loss, test_acc))

    print('***********************************************\n')

test_model(test_loader, model, criterion)
plt.figure()
plt.plot(lossval)
plt.grid()
plt.title('Loss vs iterations')
plt.show()

















