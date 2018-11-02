import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Import the mnist data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

Nodes_In_h1= 500 # n_nodes_hl1
Nodes_In_h2= 500
Nodes_In_h3= 500

Classes_no = 10  #n_classes
Size_Of_batch = 100 # batch_size

x = tf.placeholder('float', [None, 784]) # 28 * 28 =784 # placeholder creats the desired format for tensor
y = tf.placeholder('float')


#### Defining the layers 
# layer one  ===> 784 inputs and they goes to Nodes_In_h1 nodes and biases is equal to the no of nodes
# layer two  ===> Nodes_In_h1 inputs and they goes to Nodes_In_h2 nodes and biases is equal to Nodes_In_h2
####
hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, Nodes_In_h1])),
                      'biases':tf.Variable(tf.random_normal([Nodes_In_h1]))}

hidden_layer2 = {'weights':tf.Variable(tf.random_normal([Nodes_In_h1, Nodes_In_h2])),
                      'biases':tf.Variable(tf.random_normal([Nodes_In_h2]))}

hidden_layer3 = {'weights':tf.Variable(tf.random_normal([Nodes_In_h2, Nodes_In_h3])),
                      'biases':tf.Variable(tf.random_normal([Nodes_In_h3]))}
# Output layer reduces the no of inputs to no of classes
output_layer = {'weights':tf.Variable(tf.random_normal([Nodes_In_h3, Classes_no])),
                    'biases':tf.Variable(tf.random_normal([Classes_no])),}


def neural_network_tf_model(data):
    
    # BY USING FORMULA  y = Wx + b we get the output of a layer 
    Output_Of_layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['biases']) # wx + b 
    # to make it non_linear we use relu on the layer
    Output_Of_layer1  = tf.nn.relu(Output_Of_layer1 )

    Output_Of_layer2 = tf.add(tf.matmul(Output_Of_layer1,hidden_layer2['weights']), hidden_layer2['biases'])
    Output_Of_layer2 = tf.nn.relu(Output_Of_layer2)

    Output_Of_layer3 = tf.add(tf.matmul(Output_Of_layer2,hidden_layer3['weights']), hidden_layer3['biases'])
    Output_Of_layer3 = tf.nn.relu(Output_Of_layer3)

    output = tf.matmul(Output_Of_layer3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    
    prediction = neural_network_tf_model(x)
    # This cost function reduces the error 
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # optimizer optimize that further
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        # initialize the session
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/Size_Of_batch)):
                # Below function will give us the batch size x and y 
                epoch_x, epoch_y = mnist.train.next_batch(Size_Of_batch)
                # this will run the batches through the tensor flow network 
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)