import torch



def data_sampler(data, context_size, batch_size, device='cpu'):
    """
    Sample inputs and targets from dataset
    :param data: dataset
    :param context_size: size of the context window
    :param batch_size: batch size
    :return: random samples of inputs and corresponding targets
    """
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i + context_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + context_size + 1] for i in ix]).to(device)

    return x, y

def reshape_data(logits, targets):
    """
    Flatten time and batch dimension
    :param logits: logits of shape [batch_size, sequence_length, vocab_size]
    :param targets: targets of shape [batch_size, sequence_length]
    :return: logits of shape [batch_size * sequence_length, vocab_size], targets of shape [batch_size * sequence_length]
    """
    B, S, C = logits.shape
    logits = logits.view(B * S, C)
    targets = targets.view(B * S)
    return logits, targets


class Trainer():
    def __init__(self, m, train_data, val_data, vocab_size, context_size, batch_size, train_iters, eval_iters, loss_function, optimizer, learning_rate, device):
        """
        :param m: model
        :param train_data: training data
        :param val_data: validation data
        :param vocab_size: size of the vocabulary
        :param context_size: context size
        :param batch_size: batch size
        :param train_iters: training iterations
        :param eval_iters: validation iterations
        :param loss_function: loss function
        :param optimizer: optimizer
        :param learning_rate: learning rate
        """
        self.m = m
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.eval_iters = eval_iters
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = device

        # Init optimizer
        self.opt = optimizer(m.parameters(), learning_rate)



    @torch.no_grad()
    def check_loss(self):
        """
        Get current training loss and evaluation loss
        :return:
        """

        # set to evaluation mode
        self.m.eval()

        # check loss on training dataset
        train_losses = torch.zeros(self.eval_iters)
        for i in range(self.eval_iters):
            x, y = data_sampler(self.train_data, self.context_size, self.batch_size, self.device)
            logits = self.m(x)
            logits, targets = reshape_data(logits, y)
            loss = self.loss_function(logits, targets)
            train_losses[i] = loss

        mean_train_loss = torch.mean(train_losses)

        # check loss on validation dataset
        val_losses = torch.zeros(self.eval_iters)
        for i in range(self.eval_iters):
            x, y = data_sampler(self.val_data, self.context_size, self.batch_size, self.device)
            logits = self.m(x)
            logits, targets = reshape_data(logits, y)
            loss = self.loss_function(logits, targets)
            val_losses[i] = loss

        mean_val_loss = torch.mean(val_losses)

        # Set to training mode again
        self.m.train()

        return mean_train_loss, mean_val_loss


    def train(self):
        """
        train model for train_iters iterations
        """

        self.m = self.m.to(self.device)

        # training loop
        for steps in range(self.train_iters):
            x_t, y_t = data_sampler(self.train_data, self.context_size, self.batch_size, self.device)  # x_t: (batch_size, context_size)

            # forward pass
            logits = self.m(x_t)
            logits, targets = reshape_data(logits, y_t)
            loss = self.loss_function(logits, targets)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            if steps % (self.train_iters // 10) == 0:
                train_loss, val_loss = self.check_loss()
                print(f"Step {steps}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        train_loss, val_loss = self.check_loss()
        print(f"After training: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
