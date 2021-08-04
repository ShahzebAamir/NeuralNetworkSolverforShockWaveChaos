import tensorflow as tf
import numpy as np


from custom_lbfgs import lbfgs, Struct


class NeuralNetwork(object):
    def __init__(self, hp, logger, X_f, ub, lb, alpha, beta, amp, model, optimizer, x_lab=None, wn=None, sign=None):

#         layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        self.tf_epochs = hp["tf_epochs"]
        self.model = model
        self.tf_optimizer = optimizer
        self.dtype = "float64"
        self.alpha = alpha
        self.beta = beta
        self.amp = amp
        self.x_lab = x_lab
        self.wn = wn
        self.sign = sign


        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1]) #[-1,1] random points
        self.t_f = self.tensor(X_f[:, 1:2]) #[0,1] random points

#     # Computing the sizes of weights/biases for future decomposition
#     self.sizes_w = []
#     self.sizes_b = []
#     for i, width in enumerate(layers):
#         if i != 1:
#             self.sizes_w.append(int(width * layers[1]))
#             self.sizes_b.append(int(width if i != 0 else layers[1]))

        self.logger = logger

    # Defining custom loss
    # @tf.function
    #def loss(self, u, u_pred):
        #return tf.reduce_mean(tf.square(u - u_pred))

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)
    
    # @tf.function  
    # Defining custom loss
    def loss(self, u, u_pred):
        f_pred = self.f_model(self.alpha, self.beta, self.amp, self.x_lab, self.wn, self.sign)
        return tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(f_pred))
#######################################################
    def source(self, x, us, alpha, beta, amp):
        ainv = 8 * sqrt(pi * beta) * (1 + scipy.special.erf(1 / sqrt(4 * beta)))
        source = np.exp(-(x - xi(us, alpha, amp)) ** 2 / (4 * beta)) / ainv
        return source

    def flux(self, Y, us, amp, x_lab, wn, sign):
        D = Dshock(us, amp, x_lab, wn, sign)
        return 0.5 * Y ** 2. - D * Y
#######################################################
    # The actual PINN
    def f_model(self, alpha, beta, amp, x_lab, wn, sign):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        # We need the boundary conditions from the data matrix of u as well

        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            bc = np.zeros_like(self.x_f) #To get the predicted value of us at all times i.e. getting u_s
            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)
            BC_f = tf.stack([bc[:,0], self.t_f[:, 0]], axis=1) #predicted value of u at boundry at all times i.e. getting u_s

            # Getting the prediction
            u = self.model(X_f)
            us = self.model(BC_f)

            f =  self.flux(u, us, amp, x_lab, wn, sign)
            # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
            f_x = tape.gradient(f, self.x_f)

        s = self.source(self.x_f,us,alpha,beta,amp)    
        u_t = tape.gradient(u, self.t_f)

    # Letting the tape go
        del tape
    # Buidling the PINNs
        return u_t+f_x-s

    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def get_params(self, numpy=False):
        return []

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.model.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = self.tensor(w)
        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.model.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss_value = self.loss(u, self.model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def tf_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    # @tf.function
    def tf_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def nt_optimization(self, X_u, u):
        self.logger.log_train_opt("LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
    # tfp.optimizer.lbfgs_minimize(
    #   loss_and_flat_grad,
    #   initial_position=self.get_weights(),
    #   num_correction_pairs=nt_config.nCorrection,
    #   max_iterations=nt_config.maxIter,
    #   f_relative_tolerance=nt_config.tolFun,
    #   tolerance=nt_config.tolFun,
    #   parallel_iterations=6)
        self.nt_optimization_steps(loss_and_flat_grad)

    def nt_optimization_steps(self, loss_and_flat_grad):
        lbfgs(loss_and_flat_grad,
              self.get_weights(),
              self.nt_config, Struct(), True,
              lambda epoch, loss, is_iter:
              self.logger.log_train_epoch(epoch, loss, "", is_iter))

    def fit(self, X_u, u, X_star, u_star):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.tf_optimization(X_u, u)
        #self.nt_optimization(X_u, u)

        #self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

    def predict(self, X_star):
        u_star = self.model(X_star)
        f_star = self.f_model(alpha, beta, amp, x_lab, wn, sign)
        return u_star.numpy(), f_star.numpy()
                       

            


        
              

