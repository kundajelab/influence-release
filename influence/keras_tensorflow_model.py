from __future__ import absolute_import
from keras import backend as K
from .genericNeuralNet import GenericNeuralNet
import tensorflow as tf
from tensorflow.python.ops import array_ops
from influence.hessians import hessian_vector_product
import numpy as np


class KerasTensorflowModel(GenericNeuralNet):

    def __init__(self, keras_model, batch_size, data_sets,
                       model_name, **kwargs):

        self.keras_model = keras_model


        #np.random.seed(0)
        #tf.set_random_seed(0)
        
        self.batch_size = batch_size
        self.data_sets = data_sets
        log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = model_name
        self.num_classes = keras_model.output_shape[1]
        
        self.mini_batch = True
        self.damping = 0.0

        # Initialize session
        self.sess = K.get_session()
        #config = tf.ConfigProto()        
        #self.sess = tf.Session(config=config)
        #K.set_session(self.sess)
                
        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]
        
        self.logits = self.inference()

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits, 
            self.labels_placeholder)

        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)        
        self.preds = self.predictions(self.logits)

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)
    
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)

        #init = tf.global_variables_initializer()        
        #self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate([x.ravel() for x in params_val]))        
        print('Total number of parameters: %s' % self.num_params)
        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list
        return vec_to_list

    def get_all_params(self):
        return self.keras_model.weights 

    def placeholder_inputs(self):
        input_placeholder = self.keras_model.input
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels,
            K.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx],
            K.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size
    
        input_feed, labels_feed = data_set.next_batch(batch_size)                              
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,            
            K.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0            
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0            
        }
        return feed_dict

    def inference(self): 
        return self.keras_model.layers[-2].output 

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds
