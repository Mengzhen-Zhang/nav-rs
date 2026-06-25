use nalgebra::{RealField, SMatrix, SVector};
use num_dual::{Dual, Dual64, DualNum, DualNumFloat, DualVec};
use num_traits::{One, Zero};
use std::{fmt::Debug, marker::PhantomData};

pub trait Diff<T, const In: usize, const Out: usize> {
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out>;
    
    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In>;

    // jacobian vector product
    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out>;
    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>);
}

pub struct AutoDiff<T: DualNum<T>, const I: usize, const O: usize>
{
    pub f: fn(&SVector<Dual<T,T>, I>, &mut SVector<Dual<T,T>, O>)
}

impl<T: DualNum<T>, const I: usize, const O: usize> AutoDiff<T, I, O> {
    pub fn new(f: fn(&SVector<Dual<T,T>, I>, &mut SVector<Dual<T,T>, O>)) -> Self {
	Self { f }
    }
}

#[inline]
fn dual_zeros<T: DualNum<T> + From<f64>, const Out: usize>() -> SVector::<Dual<T,T>, Out> {
    SVector::<Dual<T,T>, Out>::from_element(Dual::<T,T>::from_re(T::from(0.0)))
}

impl<T: DualNum<T> + From<f64>, const In: usize, const Out: usize> Diff<T, In, Out> for AutoDiff<T, In, Out> {
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out> {
	let x_dual = x.map(|v| Dual::<T,T>::from_re(v));
	let mut y_dual = dual_zeros::<T, Out>();
	(self.f)(&x_dual, &mut y_dual);
	y_dual.map(|x| x.re)
    }

    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In> {
	let mut j = SMatrix::<T, Out, In>::zeros();
	let mut x_dual = x.map(|v| Dual::<T,T>::from_re(v));
	let mut y_dual_buffer = dual_zeros::<T,Out>();

	for i in 0..In {
	    x_dual[i].eps = T::one();
	    
	    (self.f)(&x_dual, &mut y_dual_buffer);
	    
	    j.column_mut(i)
		.copy_from(&y_dual_buffer.map(|y| y.eps));

	    x_dual[i].eps = T::zero();
	};

	j
    }

    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out> {
	let x_dual = x
	    .zip_map(v, |a,b| {
		Dual::<T,T>::new(a, b)
	    });

        let mut y_dual_buffer = dual_zeros::<T, Out>();

	(self.f)(&x_dual, &mut y_dual_buffer);

	let jvp = y_dual_buffer.map(|y| y.eps);
	jvp
    }

    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>) {
	let x_dual = x
	    .zip_map(v, |a,b| {
		Dual::<T,T>::new(a, b)
	    });
	let mut y_dual_buffer = dual_zeros::<T, Out>();

	(self.f)(&x_dual, &mut y_dual_buffer);
	let y = y_dual_buffer.map(|y| y.re);
	let jvp = y_dual_buffer.map(|y| y.eps);

	(y, jvp)
    }
}


pub struct NormalDiff<T: RealField, const In: usize, const Out: usize> {
    pub f: fn(&SVector<T, In>) -> SVector<T, Out>,
    pub jacobian: fn(&SVector<T, In>) -> SMatrix<T, Out, In>,
}

impl<T: RealField, const In: usize, const Out: usize> NormalDiff<T, In, Out> {
    pub fn new(
	f: fn(&SVector<T, In>) -> SVector<T, Out>,
	jacobian: fn(&SVector<T, In>) -> SMatrix<T, Out, In>,
    ) -> Self {
	Self { f, jacobian }
    }
}


impl<T: RealField, const In: usize, const Out: usize> Diff<T, In, Out> for NormalDiff<T, In, Out> {
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out> {
	(self.f)(x)
    }

    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In> {
	(self.jacobian)(x)
    }

    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out> {
	self.jacobian(x) * v
    }

    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>) {
	let y = self.eval(x);
	let jvp = self.jvp(x, v);
	(y, jvp)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{matrix, vector};

    // -------------------------------------------------------------------------
    // Test Case 1: Simple Linear/Affine Transformation Matrix
    // f(x, y) = [2x + 3y, 4x - y]
    // Expected Jacobian:
    // [2.0,  3.0]
    // [4.0, -1.0]
    // -------------------------------------------------------------------------
    fn linear_test_func_autodiff(x: &SVector<Dual<f64, f64>, 2>, y: &mut SVector<Dual<f64, f64>, 2>) {
        y[0] = x[0] * 2.0 + x[1] * 3.0;
        y[1] = x[0] * 4.0 - x[1];
    }

    fn linear_test_func_normal(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        vector![2.0 * x[0] + 3.0 * x[1], 4.0 * x[0] - x[1]]
    }

    fn linear_test_jacobian_normal(_x: &SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        matrix![2.0,  3.0;
                4.0, -1.0]
    }

    #[test]
    fn test_linear_autodiff() {
        let ad = AutoDiff::new(linear_test_func_autodiff);
        let x = vector![1.0, 2.0]; // Evaluation point
        let v = vector![10.0, -1.0]; // Direction vector

        let expected_eval = vector![8.0, 2.0];
        let expected_jac = matrix![2.0,  3.0;
                                   4.0, -1.0];
        // J * v = [2(10) + 3(-1), 4(10) - 1(-1)] = [17, 41]
        let expected_jvp = vector![17.0, 41.0];

        assert_eq!(ad.eval(&x), expected_eval);
        assert_eq!(ad.jacobian(&x), expected_jac);
        assert_eq!(ad.jvp(&x, &v), expected_jvp);

        let (val, jvp) = ad.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    #[test]
    fn test_linear_normal() {
        let nd = NormalDiff::new(linear_test_func_normal, linear_test_jacobian_normal);
        let x = vector![1.0, 2.0];
        let v = vector![10.0, -1.0];

        let expected_eval = vector![8.0, 2.0];
        let expected_jac = matrix![2.0,  3.0;
                                   4.0, -1.0];
        let expected_jvp = vector![17.0, 41.0];

        assert_eq!(nd.eval(&x), expected_eval);
        assert_eq!(nd.jacobian(&x), expected_jac);
        assert_eq!(nd.jvp(&x, &v), expected_jvp);

        let (val, jvp) = nd.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    // -------------------------------------------------------------------------
    // Test Case 2: Non-Linear Function (Squaring and Products)
    // f(x, y) = [x^2, x * y]
    // Expected Jacobian:
    // [2x,   0]
    // [y,    x]
    // -------------------------------------------------------------------------
    fn nonlinear_test_func_autodiff(x: &SVector<Dual<f64, f64>, 2>, y: &mut SVector<Dual<f64, f64>, 2>) {
        y[0] = x[0] * x[0];       // x^2
        y[1] = x[0] * x[1];       // x * y
    }

    #[test]
    fn test_nonlinear_autodiff() {
        let ad = AutoDiff::new(nonlinear_test_func_autodiff);
        let x = vector![3.0, 4.0];  // Evaluate at x=3, y=4
        let v = vector![2.0, 5.0];  // Direction vector

        // f(3, 4) = [9, 12]
        let expected_eval = vector![9.0, 12.0];
        
        // J = [[6, 0], [4, 3]]
        let expected_jac = matrix![6.0, 0.0;
                                   4.0, 3.0];
        
        // J * v = [[6, 0], [4, 3]] * [2, 5] = [12, 4 * 2 + 3 * 5] = [12, 23]
        let expected_jvp = vector![12.0, 23.0];

        assert_eq!(ad.eval(&x), expected_eval);
        assert_eq!(ad.jacobian(&x), expected_jac);
        assert_eq!(ad.jvp(&x, &v), expected_jvp);

        let (val, jvp) = ad.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    // -------------------------------------------------------------------------
    // Test Case 3: Polymorphism Verification via Trait Object
    // -------------------------------------------------------------------------
    #[test]
    fn test_polymorphism() {
        let ad = AutoDiff::new(linear_test_func_autodiff);
        let nd = NormalDiff::new(linear_test_func_normal, linear_test_jacobian_normal);

        let diff_engines: Vec<Box<dyn Diff<f64, 2, 2>>> = vec![Box::new(ad), Box::new(nd)];
        let x = vector![2.0, 1.0];
        let v = vector![1.0, 1.0];

        for engine in diff_engines {
            let res_eval = engine.eval(&x);
            let res_jac = engine.jacobian(&x);
            let res_jvp = engine.jvp(&x, &v);
            let (combined_val, combined_jvp) = engine.eval_jvp(&x, &v);

            assert_eq!(res_eval, vector![7.0, 7.0]);
            assert_eq!(res_jac, matrix![2.0,  3.0;
                                        4.0, -1.0]);
            // J * v = [[2, 3], [4, -1]] * [1, 1] = [5, 3]
            assert_eq!(res_jvp, vector![5.0, 3.0]);
            assert_eq!(combined_val, vector![7.0, 7.0]);
            assert_eq!(combined_jvp, vector![5.0, 3.0]);
        }
    }
}
