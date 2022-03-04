//! syron is a Rust library designed to make simple neural networks with ease.
//! # Example
//! ```rust
//! use syron::nn::{Network, Activation};
//! fn main() {
//!     // XOR Problem
//!     let x = ndarray::arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
//!     let y = ndarray::arr2(&[[0.0], [1.0], [1.0], [0.0]]);
//!
//!     // Create a network with 2 inputs and 3 hidden layers
//!     let mut nn = Network::new(2, 3, Activation::Sigmoid);
//!     nn.train(x, y, 1000);
//!     println!("{:?}", nn.predict(ndarray::arr2(&[[0.0, 0.0]])));
//! }
//! ```
pub mod nn;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use crate::nn::{Network, Activation};

        // XOR Problem
        let x = ndarray::arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = ndarray::arr2(&[[0.0], [1.0], [1.0], [0.0]]);

        // Create a network with 2 inputs and 3 hidden layers
        let mut nn = Network::new(2, 3, Activation::Sigmoid);
        nn.train(x, y, 1000);
        println!("{:?}", nn.predict(ndarray::arr2(&[[0.0, 0.0]])));
    }
}