import torch
from torch import nn

from .augmentations import GaussianSmoothing

# torch.nn.Module (neural network)
class GRUDecoder(nn.Module):
  #def __init__ to initiate classes
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        max_mask_length=25,
        num_masks=3
    ):
        # super(Class name , "self") to initialize
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        # self. to store parameters
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.max_mask_length = max_mask_length
        self.num_masks = num_masks
        # Softsign(x) = x / (1+|x|)
        self.inputLayerNonlinearity = torch.nn.Softsign()
        # (self.kernellen,1) - (height,width) is the kernel_size
        # torch.nn.Unfold extracts sliding local blocks from a batched input tensor
        # eg) 1-10, 2-11, 3-12, and so forth
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        # torch.nn.Parameter ensures dayWeights,dayBias are trainable parameters
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        # torch.eye(neural_dim) returns a 2D identity matrix with dimension neural_dim
        # .data ables us to modify tensor values without affecting gradients
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        
        # GRU (Gated recurrent unit) layers - refer to Lecture 12
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        

        """
        ###########################################################
        # Create each GRU layer separately so that we have access to intermediate layers
        self.gru_layers = nn.ModuleList()
        for i in range(layer_dim):
          input_dim = neural_dim * self.kernelLen if i == 0 else hidden_dim
          self.gru_layers.append(nn.GRU(
              input_dim,
              hidden_dim,
              batch_first=True,
              dropout=self.dropout,
              bidirectional=self.bidirectional,
          )
          )
        ############################################################
        """

        # named_parameters() returns all trainable parameters in the GRU and their corresponding names
        for name, param in self.gru_decoder.named_parameters():
            # Learnable hidden to hidden weights
            # Orthogonal -> Identity matrix
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            # Learnable input to hidden weights
            # Xavier uniform distribution U(-a,a) where a = gain x sqrt(6/(n_inputs+n_outputs))
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        # Output = self.inpLayer0 = nn.Linear(neural_dim,neural_dim)
        #          self.inpLayer1 = nn.Linear(neural_dim,neural_dim)
        #          and so on
        # torch.nn.Linear applies linear transformation to the data
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
        ###################################################
        # Layer normalization
        if self.bidirectional:
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        ###################################################

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank


    def forward(self, neuralInput, dayIdx):
        # torch.permute -> permutes the dimensions of the tensor... (a,b,c) to (0,2,1) would be (a,c,b)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        ######################################################
        # Apply time mask
        if self.training:
          neuralInput = self.apply_time_masking(neuralInput)
        ######################################################

        # apply day layer
        # 0 refers to 1st dimension (rows) for torch.index_select
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        # Batch matrix multiplication... neuralInput with shape b,t,d and dayWeights with shape b,d,k results
        # in a tensor with shape b,t,k
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        # Apply nonlinearity to ensure complexity to the model
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        # torch.unsqueeze( , 3) ensures 2D structure needed for torch.nn.Unfold
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        # device = self.device moves tensor to whatever device being used
        # .requires_grad_() ensures gradients are computed for the tensor
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        
        # self.gru_decoder = nn.GRU() defined earlier
        # h0.detach() detaches h0 from the computational graph (generates new tensor that does not require gradient)
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        ################################################
        # Apply layer normalization before output
        hid_norm = self.layer_norm(hid)
        ################################################

        # get seq
        # self.fc_decoder_out = nn.Linear() defined earlier - RNN output
        seq_out = self.fc_decoder_out(hid_norm)
        return seq_out
        
        """
        #################################################
        # Separate the first layer from the rest (different shape)
        hid, _ = self.gru_layers[0](stridedInputs, h0[0:1].detach())

        for i in range(1, self.layer_dim):
          hid, _ = self.gru_layers[i](hid, h0[i:i+1].detach())
          # We chose the 3rd layer to be the intermediate layer
          if i==2:
            hid_intermediate = hid.clone()
            hid_norm_int = self.layer_norm(hid_intermediate)
            seq_out_inter = self.fc_decoder_out(hid_norm_int)
        hid_norm = self.layer_norm(hid)
        seq_out_main = self.fc_decoder_out(hid_norm)
        #################################################
        return seq_out_main, seq_out_inter
        """

    #########################################################################
    # Define time masking function
    def apply_time_masking(self,input_tensor):
      batch_size, time_steps, channels = input_tensor.shape
      for batch in range(batch_size):
        for _ in range(self.num_masks):
          mask_length = torch.randint(1, self.max_mask_length+1, (1,)).item()
          max_start = max(1, time_steps - mask_length)
          start_time = torch.randint(0, max_start, (1,)).item()
          input_tensor[batch, start_time:start_time+mask_length, :] = 0
      return input_tensor
    ##########################################################################