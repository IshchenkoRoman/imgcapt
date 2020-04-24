import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        
        
        self._lstm = nn.LSTM(input_size=embed_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                            )
        #https://discuss.pytorch.org/t/how-should-i-understand-the-num-embeddings-and-embedding-dim-arguments-for-nn-embedding/60442
        self._emb = nn.Embedding(num_embeddings=vocab_size,
                                 embedding_dim=embed_size,
                                )
        
        self._fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
                
        captions = captions[:, :-1]
                
        captions_emb = self._emb(captions)
        features_unsqueeze = features.unsqueeze(1)
        features_unsqueeze = features_unsqueeze.float()
        
        concat = torch.cat((features_unsqueeze, captions_emb), dim=1)
        
        output, hc = self._lstm(concat)
        
        output = self._fc(output)
                        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        ans = []
        
        for _ in range(max_len):
            
            # Forward through LSTM, get LSTM outputs and cell/ memory state
            lstm_out, states = self._lstm(inputs, states)
            lstm_out = lstm_out.squeeze(1)
            
            # Get scores
            outputs = self._fc(lstm_out)
            
            # Get best predicted score 
            predicted = outputs.max(1)[1]
            if predicted.item() == 1:
                break
                                                 
            # Add predicted to answer list
            ans += [predicted.detach().cpu().item()]
            
            # Update input for next sequence
            inputs = self._emb(predicted).unsqueeze(1)
        return ans[1:]
            
    
    def init_hidden(self, batch):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch, self._hidden_size).zero_(),
                weight.new(self.n_layers, batch, self._hidden_size).zero_())
