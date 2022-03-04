use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::ArrayBase;
use ndarray::OwnedRepr;
use ndarray::Dim;
use ndarray::Array2;

/// Types of activation functions
pub enum Activation {
    Sigmoid,
}

/// The `Network` struct, used for creating and training a neural network.
pub struct Network {
    n_inputs: usize,
    n_hidden: usize,
    activation: Activation,
    weights: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>,
}

impl Network {
    /// Creates a new `Network` with the given number of inputs, hidden layers, and activation function.
    pub fn new(n_inputs: usize, n_hidden: usize, activation: Activation) -> Self {
        let mut weights = Vec::new();

        let mut w = Array::random((n_inputs, n_inputs+1), Uniform::new(-1.0, 1.0)) - 1.0;
        weights.push(w);

        for _ in 2..n_hidden {
            w = Array::random((n_inputs+1, n_inputs+1), Uniform::new(-1.0, 1.0)) - 1.0;
            weights.push(w);
        }

        w = Array::random((n_inputs+1, 1), Uniform::new(-1.0, 1.0)) - 1.0;
        weights.push(w);

        Network {
            n_inputs,
            n_hidden,
            activation,
            weights,
        }
    }

    /// Trains the network using the given inputs and outputs.
    pub fn train(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: usize) {
        let activation = match self.activation {
            Activation::Sigmoid => |x: &ndarray::Array2<f64>| x.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        };
        let activation_derivative = match self.activation {
            Activation::Sigmoid => |x: &ndarray::Array2<f64>| x.mapv(|x| x * (1.0 - x)),
        };
        // Check if number of inputs is same as self.n_inputs
        if x.shape()[1] != self.n_inputs {
            panic!("Number of inputs must be {}", self.n_inputs);
        }

        for _ in 0..epochs {
            let mut layers = vec![x.clone()];
            for i in 0..self.n_hidden {
                layers.push(activation(&layers[i].dot(&self.weights[i])));
            }

            let mut layer_errors = vec![y.clone() - &layers[layers.len()-1]];
            let mut layer_deltas = vec![layer_errors[0].clone() * activation_derivative(&layers[layers.len()-1])];

            for i in 1..self.n_hidden {
                layer_errors.push(layer_deltas[0].dot(&self.weights[self.weights.len()-1].clone().t()));
                layer_deltas.push(layer_errors[layer_errors.len()-i].clone() * activation_derivative(&layers[layers.len()-i-1]));
            }

            for (i, j) in (0..self.n_hidden).zip((0..self.n_hidden).rev()) {
                self.weights[j] += &layers[j].t().dot(&layer_deltas[i]);
            }
        }
    }

    /// Returns the output of the network given the inputs.
    pub fn predict(&self, x: Array2<f64>) -> Array2<f64> {
        let activation = match self.activation {
            Activation::Sigmoid => |x: &ndarray::Array2<f64>| x.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        };
        // Check if number of inputs is same as self.n_inputs
        if x.shape()[1] != self.n_inputs {
            panic!("Number of inputs must be {}", self.n_inputs);
        }

        let mut layers = vec![x];
        for i in 0..self.n_hidden {
            layers.push(activation(&layers[i].dot(&self.weights[i])));
        }

        layers[layers.len()-1].clone()
    }
}