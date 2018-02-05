function NeuronLayer (number_of_neurons,number_of_inputs_per_neuron) {
    this.number_of_neuronstype = number_of_neurons;
    this.number_of_inputs_per_neuron = number_of_inputs_per_neuron;
    this.synaptic_weights = makeMatrix(number_of_inputs_per_neuron,number_of_neurons);
    this.synaptic_weights = randomizeMatrix(this.synaptic_weights);
}

NeuronLayer.prototype.getInfo = function() {
    return this.synaptic_weights;
};
 
function NeuralNetwork(layer1,layer2) {
this.layer1 = layer1;
this.layer2 = layer2;
}


NeuralNetwork.prototype.train = function(training_set_inputs, training_set_outputs, number_of_training_iterations){
    for(iteration = 0; iteration < number_of_training_iterations; iteration++) {
        // Pass the training set through our neural network
        var z = this.think(training_set_inputs);
        var output_from_layer_1 = z[0];
        var output_from_layer_2 = z[1];

        // Calculate the error for layer 2 (The difference between the desired output
        // and the predicted output).
        var layer2_error = operationMatrixWithMatrix("-",training_set_outputs,output_from_layer_2);
        var layer2_delta = operationMatrixWithMatrix("*", layer2_error,normalizeMatrix(__sigmoid_derivative,output_from_layer_2));
        
        // Calculate the error for layer 1 (By looking at the weights in layer 1,
        // we can determine by how much layer 1 contributed to the error in layer 2).
        var layer1_error = dotProduct(layer2_delta,transposeMatrix(this.layer2.synaptic_weights));
        var layer1_delta = operationMatrixWithMatrix("*",layer1_error,normalizeMatrix(__sigmoid_derivative,output_from_layer_1));

        // Calculate how much to adjust the weights by
        var layer1_adjustment = dotProduct(transposeMatrix(training_set_inputs),layer1_delta);
        var layer2_adjustment = dotProduct(transposeMatrix(output_from_layer_1),layer2_delta);

        // Adjust the weights.
        this.layer1.synaptic_weights = operationMatrixWithMatrix("+",this.layer1.synaptic_weights,layer1_adjustment);
        this.layer2.synaptic_weights = operationMatrixWithMatrix("+",this.layer2.synaptic_weights,layer2_adjustment);
    }
}

NeuralNetwork.prototype.think = function(inputs) {
    var output_from_layer1 = normalizeMatrix(__sigmoid,dotProduct(inputs, this.layer1.synaptic_weights));;
    var output_from_layer2 = normalizeMatrix(__sigmoid,dotProduct(output_from_layer1, this.layer2.synaptic_weights));

    return([output_from_layer1, output_from_layer2]);

}

NeuralNetwork.prototype.print_weights = function() {
    console.log("    Layer 1 (4 neurons, each with 3 inputs): ");
    console.log(this.layer1.synaptic_weights);
    console.log("    Layer 2 (1 neuron, with 4 inputs):");
    console.log(this.layer2.synaptic_weights);
}


// Seed the random number generator
//random.seed(1)

//Create layer 1 (4 neurons, each with 3 inputs)
var layer1 = new NeuronLayer(4, 3);

// Create layer 2 (a single neuron with 4 inputs)
var layer2 = new NeuronLayer(1, 4);

layer1.synaptic_weights = [[-0.16595599,0.44064899,-0.99977125,-0.39533485],
    [-0.70648822,-0.81532281,-0.62747958,-0.30887855],
    [-0.20646505,0.07763347,-0.16161097,0.370439]];
layer2.synaptic_weights = [[-0.5910955 ],[ 0.75623487],[-0.94522481],[ 0.34093502]];


// Combine the layers to create a neural network
var neural_network = new NeuralNetwork(layer1, layer2);

console.log("Stage 1) Random starting synaptic weights: ");
neural_network.print_weights()

// The training set. We have 7 examples, each consisting of 3 input values
// and 1 output value.
var training_set_inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]];
var training_set_outputs = transposeMatrix([[0, 1, 1, 1, 1, 0, 0]]);

// Train the neural network using the training set.
// Do it 60,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 1);

console.log("Stage 2) New synaptic weights after training: ");
neural_network.print_weights();

// Test the neural network with a new situation.
console.log("Stage 3) Considering a new situation [1, 1, 0] -> ?: ");
var z = neural_network.think([1, 1, 0]);
hidden_state = z[0];
output = z[1];
console.log(output);


/*
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print "    Layer 1 (4 neurons, each with 3 inputs): "
        print self.layer1.synaptic_weights
        print "    Layer 2 (1 neuron, with 4 inputs):"
        print self.layer2.synaptic_weights

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print output
*/

/// MISC FUNCTIONS

function __sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// The derivative of the Sigmoid function.
// This is the gradient of the Sigmoid curve.
// It indicates how confident we are about the existing weight.
function __sigmoid_derivative(x) {
    return x * (1 - x);
}

function makeMatrix(I, J, fill=0.0) {
    m = [];
    for (i = 0; i < I; i++) {
        r = [];
        for (j = 0; j < J; j++) {
            r.push(fill);
        }
        m.push(r);
    }
    return m;
}

function randomizeMatrix(matrix) {
    var I = matrix.length;
    var J = matrix[0].length;
    for (i = 0; i < I; i++) {
        for (j = 0; j < J; j++) {
            matrix[i][j] = Math.random();
        }
    }
    return matrix;
}

function transposeMatrix(mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = [];

    for(j = 0; j < m; j++) {
        var r = [];
        for(i = 0; i < n; i++) {
            r.push(mat[i][j]);
        }  
        newMat.push(r);      
    }
    return(newMat);
}

function dotProduct(mat1, mat2) {
    var n1 = mat1.length;
    var m1 = mat1[0].length;

    var n2 = mat2.length;
    var m2 = mat2[0].length;

    var product = makeMatrix(n1,m2);
    
    for(i = 0; i < n1; i++) {
        for(j = 0; j < m2; j++) {
            var s = 0;
            var yyy = '';
            for(k = 0; k < m1; k++) {
                    s += mat1[i][k] * mat2[k][j];
            }  
            product[i][j] = s;
        }  
    }
    return(product);
}

function normalizeMatrix(normalizeFunction,mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            newMat[i][j] = normalizeFunction(mat[i][j]);
        }
    }
    return(newMat);    
}

function operationMatrixWithMatrix(operation = '+',mat1,mat2) {
    var n = mat1.length;
    var m = mat1[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            switch (operation) {
                case '+':
                    newMat[i][j] = mat1[i][j] + mat2[i][j];
                    break;
                case '*':
                    newMat[i][j] = mat1[i][j] * mat2[i][j];
                    break;
                case '-':
                    newMat[i][j] = mat1[i][j] - mat2[i][j];
                    break;
                case '/':
                    newMat[i][j] = mat1[i][j] / mat2[i][j];
                    break;
                default:
                    newMat[i][j] = mat1[i][j] + mat2[i][j];
                    break;
            }
        }
    }
    return(newMat);    
}

