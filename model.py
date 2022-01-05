import syft as sy

class Basic_CNN(sy.Module):
    def __init__(self, torch_ref, n_in):
        super(Basic_CNN, self).__init__(torch_ref=torch_ref)
        self.n_in = n_in
        
        self.conv1 = self.torch_ref.nn.Conv1d(1, 64, (9,), stride=1, padding=4)
        self.bn1 = self.torch_ref.nn.BatchNorm1d(64)
        self.dropout1= self.torch_ref.nn.Dropout(p=0.5)
        self.maxpool1 = self.torch_ref.nn.MaxPool1d(2,stride=2)
        
        self.conv2 = self.torch_ref.nn.Conv1d(64, 128, (5,), stride=1, padding=2)
        self.bn2 = self.torch_ref.nn.BatchNorm1d(128)
        self.dropout2 = self.torch_ref.nn.Dropout(p=0.5)
        self.avgpool = self.torch_ref.nn.AvgPool1d(2,stride=2)
        
        self.linear1 = self.torch_ref.nn.Linear(self.n_in*128 //4, 4)

        
    def forward(self, x):
        x = self.torch_ref.unsqueeze(x,1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.avgpool(x)
        
        x = self.torch_ref.flatten(x,1)
        
        return self.linear1(x)