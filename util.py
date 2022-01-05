from collections import OrderedDict

def check_cuda_capable(torch_ref, no_cuda=True, seed=42):
    has_cuda = False
    has_cuda_ptr = torch_ref.cuda.is_available()
    has_cuda = bool(has_cuda_ptr.get(
        request_block=True,
        reason="To run test and inference locally",
        timeout_secs=5,  # change to something slower
    ))
    
    use_cuda = not no_cuda and has_cuda
    torch_ref.manual_seed(seed)
    device = torch_ref.device("cuda" if use_cuda else "cpu")
    print(f"Data Owner device is {device.type.get()}")
    
    return use_cuda

def train(model, client, train_loader, epoch, args, train_data_length, batch_size=1, log_interval=10, use_cuda=False, dry_run=False):
    
    torch_ref = client.torch
    
    # initialize optimzer
    optimizer = torch_ref.optim.Adam(model.parameters(), lr=args["lr"])
    
    # + 0.5 lets us math.ceil without the import
    train_batches = round((train_data_length / batch_size) + 0.5)
    print(f"\t> Running train in {train_batches} batches")
    
    if model.is_local:
        print("\tTraining requires remote model")
        return

    model.train()

    for batch_idx, data in enumerate(train_loader):
        data_ptr, target_ptr = data[0], data[1]
        if use_cuda:
            data_ptr, target_ptr = data_ptr.cuda(), target_ptr.cuda()
        optimizer.zero_grad()
        output = model(data_ptr)
        loss = torch_ref.nn.functional.cross_entropy(output, target_ptr)
        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        train_loss = client.python.Float(0)  # create a remote Float we can use for summation
        train_loss += loss_item
        if batch_idx % log_interval == 0:
            local_loss = None
            local_loss = loss_item.get(
                reason="To evaluate training progress",
                request_block=True,
                timeout_secs=5
            )
            if local_loss is not None:
                print("\tTrain Epoch: {} {} {:.4}".format(epoch, batch_idx, local_loss))
            else:
                print("\tTrain Epoch: {} {} ?".format(epoch, batch_idx))
        if batch_idx >= train_batches - 1:
            print("\tbatch_idx >= train_batches, breaking")
            break
        if dry_run:
            break
        
        
def test_local(model, torch_ref, test_loader, test_data_length, batch_size =1, dry_run=False):
    # download remote model
    if not model.is_local:
        local_model = model.get(
            request_block=True,
            reason="test evaluation",
            timeout_secs=5
        )
    else:
        local_model = model
        
    # + 0.5 lets us math.ceil without the import
    test_batches = round((test_data_length / batch_size) + 0.5)
    print(f"> Running test_local in {test_batches} batches")
    local_model.eval()
    test_loss = 0.0
    correct = 0.0

    with torch_ref.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = local_model(data)
            iter_loss = torch_ref.nn.functional.cross_entropy(output, target, reduction="mean").item()
            test_loss = test_loss + iter_loss
            pred = output.argmax(dim=1)
            total = pred.eq(target).sum().item()
            correct += total
            if dry_run:
                break
                
            if batch_idx >= test_batches - 1:
                print("batch_idx >= test_batches, breaking")
                break

    accuracy = correct / test_data_length
    print(f"Test Set Accuracy: {100 * accuracy}%")
    
    
def update_gradient(global_model,all_model, all_data_length):
    print(f'Averaging updates to global model from {len(all_model)} clients')
    # count contributions
    total_data = sum(all_data_length)
    coefficients = [data / total_data for data in all_data_length]
    
    averaged_weights = OrderedDict()
    for idx, client_model in enumerate(all_model):
        
        local_weights = client_model.get(
            reason="To evaluate training progress",
            timeout_secs=5,
            request_block=True).state_dict()
        
        for key in global_model.state_dict().keys():
            if idx == 0:
                averaged_weights[key] = coefficients[idx] * local_weights[key]
            else:
                averaged_weights[key] += coefficients[idx] * local_weights[key]
    
    # update global model
    global_model.load_state_dict(averaged_weights)
    
def get_train_length(client, storage_index):
    train_data_length = len(client.store[storage_index])
    return train_data_length

def get_dataloader(client, storage_index, batch_size):
    return client.torch.utils.data.DataLoader(client.store[storage_index], batch_size=batch_size)