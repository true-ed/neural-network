#include <iostream>;
#include <vector>;
#include <random>;
#include <chrono>;

using namespace std;
using Matrix = vector<vector<double>>;

uniform_real_distribution<double> unif(-1, 1);
default_random_engine random;

void sigmoid(double& x) {
	x = (1 / (1 + exp(-x)));
}

void derivative_sigmoid(double& x) {
	sigmoid(x);
	x = (x * (1 - x));
}

void gen_matrix(Matrix& matrix,
	const int& x,
	const int& y) {
	for (int _x = 0; _x < x; _x++)
	{
		matrix.push_back({});
		for (int _y = 0; _y < y; _y++)
		{
			matrix[_x].push_back(unif(random));
		}
	}
}

void gen_matrix(Matrix& matrix,
	const int& x,
	const int& y,
	const double& value) {
	for (int _x = 0; _x < x; _x++)
	{
		matrix.push_back({});

		for (int _y = 0; _y < y; _y++)
		{
			matrix[_x].push_back(value);
		}
	}
}

class Layer {
public:
	int size;

	Layer* incoming_layer = NULL;
	Layer* outgoing_layer = NULL;

	Matrix weights;

	Matrix outputs;
	Matrix derivative_outputs;
	Matrix errors;

	Matrix bias;

	bool is_copy = false;

	Layer(const int& _size) : size(_size) {
		gen_matrix(outputs, size, 1, 0);
		gen_matrix(derivative_outputs, size, 1, 1);
		gen_matrix(errors, size, 1, 0);
		gen_matrix(bias, size, 1);
	}

	Layer(const Layer& layer) {

		size = layer.size;
		weights = layer.weights;
		outputs = layer.outputs;
		bias = layer.bias;

		gen_matrix(derivative_outputs, size, 1, 1);
		gen_matrix(errors, size, 1, 0);

		is_copy = true;
	}

	void Connect(Layer& const _outgoing_layer) {
		outgoing_layer = &_outgoing_layer;
		outgoing_layer->incoming_layer = this;

		if (!is_copy) {
			gen_matrix(weights, size, outgoing_layer->size);
		}
	}

	inline Matrix Activation(const Matrix& inputs) {
		if (!incoming_layer) {
			outputs = inputs;

			return outgoing_layer->Activation(outputs);
		}

		Matrix sum = bias;

		for (int i = 0; i < size; i++)
		{
			for (int k = 0; k < incoming_layer->size; k++)
			{
				sum[i][0] += incoming_layer->outputs[k][0] * \
					incoming_layer->weights[k][i];
			}
		}

		outputs = sum;
		derivative_outputs = sum;

		for (int i = 0; i < size; i++)
		{
			sigmoid(outputs[i][0]);
			derivative_sigmoid(derivative_outputs[i][0]);
		}

		if (outgoing_layer) {
			return outgoing_layer->Activation(outputs);
		}

		return outputs;
	}

	void Propagate(const Matrix& target, const double& rate = 0.3) {
		Matrix sum;

		for (int i = 0; i < size; i++)
		{
			sum.push_back({ 0 });
			sum[i][0] = outputs[i][0] - target[i][0];

			errors[i][0] = sum[i][0] * derivative_outputs[i][0];
			bias[i][0] -= rate * errors[i][0];
		}

		incoming_layer->Propagate(rate);
	}

	void Propagate(const double& rate = 0.3) {
		Matrix sum;

		for (int i = 0; i < size; i++)
		{
			sum.push_back({ 0 });

			for (int k = 0; k < outgoing_layer->size; k++)
			{
				weights[i][k] -= rate * outgoing_layer->errors[k][0] \
					* outputs[i][0];

				sum[i][0] += weights[i][k] * outgoing_layer->errors[k][0];
			}

			errors[i][0] = sum[i][0] * derivative_outputs[i][0];
			bias[i][0] -= rate * errors[i][0];
		}

		if (incoming_layer) {
			incoming_layer->Propagate(rate);
		}
	}
};

int main() {

	// Example Part

	double rate = 0.8;

	Layer input_layer(2);
	Layer hidden_layer_1(10);
	Layer output_layer(1);

	input_layer.Connect(hidden_layer_1);
	hidden_layer_1.Connect(output_layer);

	cout << "XOR example: " << endl;

	auto training_time_begin = std::chrono::steady_clock::now();

	for (int i = 0; i < 1000; i++) {
		input_layer.Activation({ {0}, {0} });
		output_layer.Propagate({ {0} }, rate);

		input_layer.Activation({ {1}, {0} });
		output_layer.Propagate({ {1} }, rate);

		input_layer.Activation({ {0}, {1} });
		output_layer.Propagate({ {1} }, rate);

		input_layer.Activation({ {1}, {1} });
		output_layer.Propagate({ {0} }, rate);
	}

	auto training_time_end = std::chrono::steady_clock::now();
	auto training_time = std::chrono:: \
		duration_cast<std::chrono::milliseconds> \
		(training_time_end - training_time_begin) \
		.count();

	cout << "Neural network training time: ";
	cout << training_time << endl;

	cout << "0 and 0 to 0 : ";
	cout << input_layer.Activation({ {0}, {0} })[0][0] << endl;

	cout << "1 and 0 to 0 : ";
	cout << input_layer.Activation({ {1}, {0} })[0][0] << endl;

	cout << "0 and 1 to 0 : ";
	cout << input_layer.Activation({ {0}, {1} })[0][0] << endl;

	cout << "1 and 1 to 1 : ";
	cout << input_layer.Activation({ {1}, {1} })[0][0] << endl;

	return 0;
}