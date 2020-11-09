let sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};

let derivative_sigmoid = (x) => {
  return sigmoid(x) * (1 - sigmoid(x));
};

let gen_matrix = (x, y, value = false) => {
  let matrix = [];

  for (let x_i = 0; x_i < x; x_i++) {
    matrix.push([]);

    for (let y_i = 0; y_i < y; y_i++) {
      value === false
        ? matrix[x_i].push(Math.random() * 2 - 1)
        : matrix[x_i].push(value());
    }
  }

  return matrix;
};

class Layer {
  constructor(size, rate = 0.3) {
    this.incoming_layer = false;
    this.outgoing_layer = false;

    this.size = size;
    this.rate = rate;

    this.outputs = gen_matrix(size, 1, () => 0);
    this.derivative_outputs = gen_matrix(size, 1, () => 1);
    this.errors = gen_matrix(size, 1, () => 0);
    this.bias = gen_matrix(size, 1);
  }

  Connect(outgoing_layer) {
    this.outgoing_layer = outgoing_layer;
    this.outgoing_layer.incoming_layer = this;
    this.weights = gen_matrix(this.size, this.outgoing_layer.size);
  }

  Activation(inputs = false) {
    if (!this.incoming_layer) {
      this.outputs = inputs;

      return this.outgoing_layer.Activation(this.outputs);
    }

    let sum = gen_matrix(this.size, 1);

    for (let i = 0; i < this.size; i++) {
      for (let k = 0; k < this.incoming_layer.size; k++) {
        sum[i][0] +=
          this.incoming_layer.outputs[k][0] * this.incoming_layer.weights[k][i];
      }
    }

    for (let i = 0; i < this.size; i++) {
      this.outputs[i][0] = sigmoid(sum[i][0]);
      this.derivative_outputs[i][0] = derivative_sigmoid(sum[i][0]);
    }

    if (this.outgoing_layer) {
      return this.outgoing_layer.Activation(this.outputs);
    }

    return this.outputs;
  }

  Propagate(outputs = []) {
    let sum = gen_matrix(this.size, 1, () => 0);

    if (this.outgoing_layer) {
      for (let i = 0; i < this.size; i++) {
        for (let k = 0; k < this.outgoing_layer.size; k++) {
          this.weights[i][k] -=
            this.rate * this.outgoing_layer.errors[k][0] * this.outputs[i][0];

          sum[i][0] += this.weights[i][k] * this.outgoing_layer.errors[k][0];
        }
      }
    } else {
      for (let i = 0; i < this.size; i++) {
        sum[i][0] = this.outputs[i][0] - outputs[i][0];
      }
    }

    for (let i = 0; i < this.size; i++) {
      this.errors[i][0] = sum[i][0] * this.derivative_outputs[i][0];
      this.bias[i][0] -= this.rate * this.errors[i][0];
    }

    if (this.incoming_layer) {
      this.incoming_layer.Propagate();
    }
  }
}

let rate = 0.8;

let input_layer = new Layer(2, rate);
let hidden_layer_1 = new Layer(10, rate);
let output_layer = new Layer(1, rate);

input_layer.Connect(hidden_layer_1);
hidden_layer_1.Connect(output_layer);

console.log("XOR example:");

let training_time_begin = new Date();

for (let i = 0; i < 1000; i++) {
  input_layer.Activation([[0], [0]]);
  output_layer.Propagate([[0]]);

  input_layer.Activation([[1], [0]]);
  output_layer.Propagate([[1]]);

  input_layer.Activation([[0], [1]]);
  output_layer.Propagate([[1]]);

  input_layer.Activation([[1], [1]]);
  output_layer.Propagate([[0]]);
}

let training_time_end = new Date();

let training_time = Math.abs(training_time_end - training_time_begin);

console.log(`Neural network training time: ${training_time}`);

console.log(`0 and 0 to 0 : ${input_layer.Activation([[0], [0]])}`);
console.log(`0 and 1 to 1 : ${input_layer.Activation([[0], [1]])}`);
console.log(`1 and 0 to 1 : ${input_layer.Activation([[1], [0]])}`);
console.log(`1 and 1 to 0 : ${input_layer.Activation([[1], [1]])}`);
