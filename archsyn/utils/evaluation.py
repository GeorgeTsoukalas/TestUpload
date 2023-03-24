import torch
from sklearn.metrics import hamming_loss, f1_score

def compute_average_f1_score(predicted, truth, num_labels):
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(truth, torch.Tensor)

    if num_labels > 1:
        weighted_avg_f1 = f1_score(truth, predicted, average='weighted')
        unweighted_avg_f1 = f1_score(truth, predicted, average='macro')
        all_f1 = f1_score(truth, predicted, average=None)
        return weighted_avg_f1, unweighted_avg_f1, all_f1
    else:
        avg_f1 = f1_score(truth, predicted, average='binary')
        all_f1 = f1_score(truth, predicted, average=None)
        return avg_f1, all_f1

def label_correctness(predictions, truths, num_labels=1, threshold=0.5):
    #counts up hamming distance and true accuracy
    additional_scores = {}
    if len(predictions.size()) == 1:
        predictions = torch.sigmoid(predictions) > threshold
    else:
        assert len(predictions.size()) == 2
        predictions = torch.max(predictions, dim=-1)[1]

    additional_scores['hamming_accuracy'] = 1 - hamming_loss(truths.squeeze().cpu(), predictions.squeeze().cpu())
    if num_labels > 1:
        w_avg_f1, additional_scores['unweighted_f1'], additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores
    else:
        w_avg_f1, additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores

Testing = True

def lossfxn(out, labels, device):
        if Testing:
            out.to(device)
            
            augmented_out = [(torch.ones(1).to(device) - out[l][1]) if labels[l] == 1 else ( out[l][0] if labels[l] == 2 else (torch.ones(1).to(device) - (torch.ones(1).to(device) - out[l][0])*out[l][1])) for l in range(len(out))] # added implication example here (assuming p-> q atm)
            return torch.mean(torch.hstack(augmented_out))
            augmented_out = [out[l][1][0] if labels[l] == 2 else out[l][0][0] for l in range(len(out))]
            return torch.mean(torch.hstack(augmented_out))
            if len(out[1]) == 0:
                return torch.mean(torch.square(torch.prod(out[0][0])))
            elif len(out[0]) == 0:
                return torch.mean(torch.square(torch.prod(torch.ones(len(out[1])) - out[1][0]))) # again this only works for size 1 at the moment
            else:
                return torch.mean(torch.square(torch.prod(out[0][0]) * torch.prod(torch.ones(len(out[1])) - out[1][0]))) # modify maybe the empty product fucks up the gradient
        else:
            return torch.nn.MSELoss()(out, labels)

def value_correctness(predictions, truths, device, num_labels=1, threshold=0.5): #Fixme. Num_labels and threshold not important here.
    with torch.no_grad():
        return lossfxn(predictions, truths, device), None
