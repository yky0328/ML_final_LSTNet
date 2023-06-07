import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

import argparse
from utils.load_data import *
from utils.optim import *
from models.DeepModels import *
from models.Attention import *
from utils.metrics import *

params={
    'data': '../dataset/commodity.npy',
    'horizon': 1,
    'window': 32,
    'highway_window': 14,
    'skip': -1,
    'model': 'LSTNet',
    'CNN_kernel': 2,
    'hidRNN': 50,
    'hidCNN': 50,
    'hidSkip': 0,
    'L1Loss': False,
    'epochs': 150,
    'batch_size': 64,
    'output_fun': 'linear',
    'dropout': 0.2,
    'save': '../save/commodity_LSTNet.pt',
    'clip': 10,
    'seed': 12345,
    'log_interval': 2000,
    'optim': 'adam',
    'lr': 0.001,
    'normalize': 2,
    'cuda': 0,
    'gpu': 0,
    'variables': 20,
    'input_dim': 137,
    'sci_kernel_size': 5,
    'sci_hidden_size': 1

}

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds'):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        # print(X.shape)
        # print(Y.shape)
        output = model(X)

        scale = data.scale.expand(output.size(0), data.m)
        # total_loss += evaluateL2(output * scale, Y * scale).data[0]
        # total_loss_l1 += evaluateL1(output * scale, Y * scale).data[0]
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

        if predict is None:
            predict = output * scale
            test = Y * scale
        else:
            predict = torch.cat((predict, output * scale))
            test = torch.cat((test, Y * scale))

    rse = math.sqrt(total_loss / n_samples) / data.rse
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    v = Ytest.reshape(-1)
    v_ = predict.reshape(-1)
    mae = MAE(v, v_)
    mape = MAPE(v, v_)
    rmse = RMSE(v, v_)
    smape = sMAPE(v, v_)
    smape2 = sMAPE2(v, v_)

    return rse, rmse, mape, mae, smape, smape2


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        # total_loss += loss.data[0]
        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


def ModelTest(params):
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default=params['data'],
                        help='location of the data file')
    parser.add_argument('--model', type=str, default=params['model'],
                        help='')
    parser.add_argument('--hidCNN', type=int, default=params['hidCNN'],
                        help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=params['hidRNN'],
                        help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=params['window'],
                        help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=params['CNN_kernel'],
                        help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=params['highway_window'],
                        help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=params['epochs'],
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=params['dropout'],
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=params['gpu'])
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default=params['save'],
                        help='path to save the final model')
    parser.add_argument('--cuda', type=str, default=params['cuda'])
    parser.add_argument('--optim', type=str, default=params['optim'])
    parser.add_argument('--lr', type=float, default=params['lr'])
    parser.add_argument('--horizon', type=int, default=params['horizon'])
    parser.add_argument('--skip', type=float, default=params['skip'])
    parser.add_argument('--hidSkip', type=int, default=params['hidSkip'])
    parser.add_argument('--L1Loss', type=bool, default=params['L1Loss'])
    parser.add_argument('--normalize', type=int, default=params['normalize'])
    parser.add_argument('--output_fun', type=str, default=params['output_fun'])

    # sci block
    parser.add_argument('--input_dim', type=int, default=params['input_dim'])
    parser.add_argument('--sci_kernel_size', type=int, default=params['sci_kernel_size'])
    parser.add_argument('--sci_hidden_size', type=int, default=params['sci_hidden_size'])

    args = parser.parse_args()

    Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'],
                        params['normalize'])
    print(Data.rse)
    print(Data.dat)
    print(args.window)
    model = eval(params['model'])(args, Data)

    if params['cuda']:
        model.to('mps')


    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if params['L1Loss']:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if params['cuda']:
        # criterion = criterion.cuda()
        # evaluateL1 = evaluateL1.cuda()
        # evaluateL2 = evaluateL2.cuda()

        criterion = criterion.to('mps')
        evaluateL1 = evaluateL1.to('mps')
        evaluateL2 = evaluateL2.to('mps')

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, params['epochs'] + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
            val_rse, val_rmse, val_mape, val_mae, val_smape, val_smape2 = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2,
                                                            evaluateL1,
                                                            params['batch_size'])
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rsme {:5.4f} | '
                'valid mape  {:5.4f} | valid mae  {:5.4f} | valid smape  {:5.4f} | valid smape2  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rmse, val_mape, val_mae, val_smape, val_smape2))
            # Save the model if the validation loss is the best we've seen so far.

            if val_rse < best_val:
                with open(params['save'], 'wb') as f:
                    torch.save(model, f)
                best_val = val_rse
            if epoch % 5 == 0:
                test_rse, test_rmse, test_mape, test_mae, test_smape, test_smape2 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                    evaluateL1,
                                                                    params['batch_size'])
                print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f} | test smape {:5.4f} | test smape2 {:5.4f}".format(test_rse,
                                                                                                           test_rmse,
                                                                                                           test_mape,
                                                                                                           test_mae,
                                                                                                           test_smape,
                                                                                                           test_smape2))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(params['save'], 'rb') as f:
        model = torch.load(f)
    test_rse, test_rmse, test_mape, test_mae, test_smape, test_smape2 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                        params['batch_size'])
    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f} | test smape  {:5.4f} | test smape2 {:5.4f}".format(test_rse, test_rmse,
                                                                                                test_mape, test_mae, test_smape, test_smape2))

def run_model(params):
    Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'],
                        params['normalize'])
    with open(params['save'], 'rb') as f:
        model = torch.load(f)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    test_rse, test_rmse, test_mape, test_mae, test_smape, test_smape2 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                        params['batch_size'])
    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f} | test smape  {:5.4f} | test smape2 {:5.4f}".format(test_rse, test_rmse,
                                                                                                test_mape, test_mae, test_smape, test_smape2))

def main():

    params['data'] = '../dataset/solar_AL.txt'
    params['model'] = 'LSTNet'
    params['save'] = '../save/solar_AL_horizon3_skip24_sci.pt'
    params['horizon'] = 3
    params['skip'] = 24
    params['hidSkip'] = 10
    params['batch_size'] = 64
    ModelTest(params)
    # run_model(params)

    # params['data'] = '../dataset/traffic.txt'
    # params['model'] = 'LSTNet'
    # params['save'] = '../save/traffic_LSTNet_horizon6_skip24_ssa.pt'
    # params['horizon'] = 6
    # params['skip'] = 24
    # params['hidSkip'] = 10
    # ModelTest(params)
    # run_model(params)

    # params['data'] = '../dataset/traffic.txt'
    # params['model'] = 'LSTNet'
    # params['save'] = '../save/traffic_LSTNet_horizon6.pt'
    # params['horizon'] = 6
    # run_model(params)

    # params['data'] = '../dataset/traffic.txt'
    # params['model'] = 'LSTNet'
    # params['save'] = '../save/traffic_LSTNet.pt'
    # run_model(params)

main()