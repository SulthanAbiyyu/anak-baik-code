import torch
import torch.nn as nn


class BLEU:
    def __init__(self, n_grams=4):
        self.n_grams = n_grams

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.split()
        y_true = y_true.split()

        n_grams = self.n_grams
        pred_n_grams = [y_pred[i:i + n_grams] for i in range(len(y_pred) - n_grams + 1)]
        true_n_grams = [y_true[i:i + n_grams] for i in range(len(y_true) - n_grams + 1)]

        pred_n_grams = {tuple(n_gram) for n_gram in pred_n_grams}
        true_n_grams = {tuple(n_gram) for n_gram in true_n_grams}

        n_correct = len(pred_n_grams.intersection(true_n_grams))
        n_total = len(pred_n_grams)

        return n_correct / n_total
    
class ROGUEN:
    def __init__(self, n_grams=4):
        self.n_grams = n_grams

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.split()
        y_true = y_true.split()

        n_grams = self.n_grams
        pred_n_grams = [y_pred[i:i + n_grams] for i in range(len(y_pred) - n_grams + 1)]
        true_n_grams = [y_true[i:i + n_grams] for i in range(len(y_true) - n_grams + 1)]

        pred_n_grams = {tuple(n_gram) for n_gram in pred_n_grams}
        true_n_grams = {tuple(n_gram) for n_gram in true_n_grams}

        n_correct = len(pred_n_grams.intersection(true_n_grams))
        n_total = len(true_n_grams)

        return n_correct / n_total

class ROGUEL:
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        y_pred = y_pred.split()
        y_true = y_true.split()
        m, n = len(y_pred), len(y_true)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if y_pred[i - 1] == y_true[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n] / n
    
def test_BLEU():
    bleu = BLEU()
    y_pred = "the cat in the hat"
    y_true = "the cat sat on the mat"
    print(bleu(y_pred, y_true)) 

def test_ROGUEN():
    roguen = ROGUEN()
    y_pred = "the cat in the hat"
    y_true = "the cat sat on the mat"
    print(roguen(y_pred, y_true))  
    
def test_ROGUEL():
    roguen = ROGUEL()
    y_pred = "the cat in the hat"
    y_true = "the cat sat on the mat"
    print(roguen(y_pred, y_true))  

if __name__ == "__main__":
    test_BLEU()
    test_ROGUEN()
    test_ROGUEL()