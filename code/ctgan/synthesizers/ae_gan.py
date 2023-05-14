import warnings
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.ctgan import Discriminator, Generator
from ctgan.synthesizers.encoders.autoencoder import AutoEncoder
from ctgan.synthesizers.encoders.entity_embeddings_ae import EntityEmbeddingEncoder
from ctgan.synthesizers.encoders.vae import VariationalAutoEncoder

def get_st_ed(target_col_index,output_info):
    
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 
    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) output_info -> column information corresponding to the data after applying the data transformer
    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c= 0
    # counter to iterate through column information
    tc= 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    
    
    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed= st+output_info[tc][0] 
    
    return (st,ed)


class Classifier(nn.Module):
    """
    This class represents the classifier module used along side the discriminator to train the generator network
    Variables:
    1) dim -> column dimensionality of the transformed input data after removing target column
    2) class_dims -> list of dimensions used for the hidden layers of the classifier network
    3) str_end -> tuple containing the starting and ending positions of the target column in the transformed input data
    Methods:
    1) __init__() -> initializes and builds the layers of the classifier module 
    2) forward() -> executes the forward pass of the classifier module on the corresponding input data and
                    outputs the predictions and corresponding true labels for the target column 
    """
    
    def __init__(self,input_dim, class_dims,st_ed):
        super(Classifier,self).__init__()
        # subtracting the target column size from the input dimensionality 
        self.dim = input_dim-(st_ed[1]-st_ed[0])
        # storing the starting and ending positons of the target column in the input data
        self.str_end = st_ed
        
        # building the layers of the network with same hidden layers as discriminator
        seq = []
        tmp_dim = self.dim

        for item in list(class_dims):
            seq += [
                nn.Linear(tmp_dim, item),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ]
            tmp_dim = item
        
        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1]-st_ed[0])==2:
            seq += [nn.Linear(tmp_dim, 1), nn.Sigmoid()]
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class
        else: 
            seq += [nn.Linear(tmp_dim, (st_ed[1]-st_ed[0]))] 
            
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        # true labels obtained from the input data
        label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        # input to be fed to the classifier module
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        # returning predictions and true labels for binary/multi-class classification 
        if ((self.str_end[1]-self.str_end[0])==2):
            return self.seq(new_imp).view(-1), label
        else: 
            return self.seq(new_imp), label


class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGANV2(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                 ae_dim=(256, 128, 64), clf_dim=(256, 256, 256, 256), generator_lr=2e-4, generator_decay=1e-6, 
                 discriminator_lr=2e-4, discriminator_decay=1e-6, autoencoder_lr=1e-4, clf_lr=2e-4, 
                 clf_betas=(0.5, 0.9), clf_eps=1e-3, clf_decay=1e-5, batch_size=500, ae_batch_size=512, 
                 discriminator_steps=1, log_frequency=True, verbose=False, epochs=300, ae_epochs=100, pac=10, cuda=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._autoencoder_dim = ae_dim
        self._clf_dim = clf_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._autoencoder_lr = autoencoder_lr
        self._clf_optim_params = dict(lr=clf_lr, betas=clf_betas, eps=clf_eps, weight_decay=clf_decay)

        self._batch_size = batch_size
        self._ae_batch_size = ae_batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._ae_epochs = ae_epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'
            
        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._autoencoder = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = nn.functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None, ae_epochs=None, target_index=None, dt=None):
        """Fit the CTGAN Synthesizer models to the training data.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        # epochs for GAN
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        # epochs for AE
        if ae_epochs is None:
            ae_epochs = self._ae_epochs

        # transform data - MSN + OH
        if dt is not None:
            self._transformer = dt
        else:
            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)
        
        # self._autoencoder = AutoEncoder(
        #     input_dim=train_data.shape[1], 
        #     hidden_dims=self._autoencoder_dim
        # ).to(self._device)

        # self._autoencoder = VariationalAutoEncoder(
        #     input_dim=train_data.shape[1], 
        #     hidden_dims=self._autoencoder_dim
        # ).to(self._device)

        self._autoencoder = EntityEmbeddingEncoder(
            input_dim=train_data.shape[1],
            hidden_dims=self._autoencoder_dim,
            output_info=self._transformer.output_info_list
        ).to(self._device)

        # ae training setup
        optimizerAE = torch.optim.Adam(self._autoencoder.parameters(), lr=self._autoencoder_lr)
        ae_loss_fn = nn.MSELoss()
        self._ae_losses = np.zeros(ae_epochs)
        dataloader = DataLoader(train_data, batch_size=self._ae_batch_size)

        for it in range(ae_epochs):
            it_loss = 0
            
            for batch in dataloader:
                # forward pass
                batch = batch.to(self._device, dtype=torch.float32)
                out = self._autoencoder(batch)                
                loss = ae_loss_fn(out, batch)
                
                if isinstance(self._autoencoder, VariationalAutoEncoder):
                    loss += self._autoencoder.kld

                # backprop
                optimizerAE.zero_grad()
                loss.backward()
                optimizerAE.step()
                it_loss += loss.item()
            
            if self._verbose and it % 10 == 0:
                print(f'AE training epoch {it} MSE: {it_loss}')
            
            self._ae_losses[it] = it_loss
        
        data_dim = self._autoencoder_dim[-1]

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        # classification module
        if target_index is not None:
            # determine st_ed indices
            output_info = [item for sub_list in self._transformer.output_info_list for item in sub_list]
            st_ed = get_st_ed(target_index, output_info)
            
            classifier = Classifier(train_data.shape[1], self._clf_dim, st_ed).to(self._device)
            
            optimizerC = torch.optim.Adam(
                classifier.parameters(), 
                **self._clf_optim_params
            )

        optimizerG = torch.optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = torch.optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        self._d_losses = np.zeros(epochs)
        self._g_losses = np.zeros(epochs)
        self._c_losses = np.zeros(epochs)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                real = None

                ## discriminator training
                for n in range(self._discriminator_steps):
                    # input generation for generator
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    # generator forward pass
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    # real input for discriminator
                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    real_enc = self._autoencoder.encode(real)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real_enc, c2], dim=1)
                    else:
                        real_cat = real_enc
                        fake_cat = fakeact

                    # discriminator forward pass
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    # discriminator loss
                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    # discriminator backprop
                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                ## generator training
                # input generation for generator
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                # generator forward pass
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # ce between fake and cond vec
                if condvec is None:
                    cross_entropy = 0
                else:
                    fake_decode = self._autoencoder.decode(fakeact)
                    cross_entropy = self._cond_loss(fake_decode, c1, m1)
                
                # generator loss
                loss_g = -torch.mean(y_fake) + cross_entropy

                # generator backprop
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

                ## clf training
                if target_index is not None:
                    clf_loss_fn = nn.BCELoss() if st_ed[1] - st_ed[0] == 2 else nn.CrossEntropyLoss()
                    
                    # clf forward pass on real data
                    real_pre, real_label = classifier(real)

                    if (st_ed[1] - st_ed[0])==2:
                        real_label = real_label.type_as(real_pre)

                    # clf loss
                    loss_cr = clf_loss_fn(real_pre, real_label)

                    # clf backprop on real data
                    optimizerC.zero_grad()
                    loss_cr.backward()
                    optimizerC.step()

                    # generator forward pass
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    fake_dec = self._autoencoder.decode(fakeact)

                    # clf forward pass on fake data
                    fake_pre, fake_label = classifier(fake_dec)

                    if (st_ed[1] - st_ed[0])==2:
                        fake_label = fake_label.type_as(fake_pre)

                    # clf loss
                    loss_cf = clf_loss_fn(fake_pre, fake_label)

                    # clf backprop on fake data
                    optimizerC.zero_grad()
                    loss_cf.backward()
                    optimizerC.step()

            self._d_losses[i] = loss_d.item()
            self._g_losses[i] = loss_g.item()
            self._c_losses[i] = loss_cr.item() + loss_cf.item()

            if self._verbose:
                msg = f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f}, Loss D: {loss_d.detach().cpu(): .4f}'

                if target_index is not None:
                    msg += f'Loss C: {loss_cr.item() + loss_cf.item():.4f}'

                print(msg, flush=True)

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []

        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            fake_dec = self._autoencoder.decode(fakeact)
            data.append(fake_dec.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
            