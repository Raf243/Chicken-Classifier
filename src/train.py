from utils import *

def train(X, Y, layers_dims, learning_rate = 0.0075, iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters(layers_dims)
    # Loop (gradient descent)
    for i in range(0, iterations):

        AL, caches = forward_prop(X, parameters)
        
        cost = compute_cost(AL, Y)
        
        grads = backward_prop(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs

