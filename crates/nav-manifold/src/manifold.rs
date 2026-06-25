use nalgebra::{RealField, SMatrix, SVector, Scalar};
use num_traits::Zero;

pub trait Manifold<S, const A_DIM: usize, const T_DIM: usize> {
    type TangentVector;

    // x_next = x ⊕ δx
    fn retract(&self, delta: &Self::TangentVector) -> Self;

    // δx = x₂ ⊖ x₁
    fn local_lift(&self, other: &Self) -> Self::TangentVector;

    // TₓM → TₓA
    fn pushforward_jacobian(&self) -> SMatrix<S, A_DIM, T_DIM>;

    // jacobian vector product
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, A_DIM>;

    fn vector_to_tangent(vec: &SVector<S, T_DIM>) -> Self::TangentVector;

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, T_DIM>;
}

pub trait InvariantLieGroup<S, const A_DIM: usize, const T_DIM: usize>: Manifold<S, A_DIM, T_DIM> {
    // lie exp
    fn exp(omega: &Self::TangentVector) -> Self;

    // lie log
    fn log(&self) -> Self::TangentVector;

    // lie group multiplication
    fn compose(&self, other: &Self) -> Self;

    // lie group inverse
    fn inverse(&self) -> Self;

    // Ad_g
    fn adjoint(&self) -> SMatrix<S, T_DIM, T_DIM>;

    // ad_omega
    fn small_adjoint(omega: &Self::TangentVector) -> SMatrix<S, T_DIM, T_DIM>;
}



#[derive(Clone, Debug)]
pub struct ProductSpace<M1, M2, const A1: usize, const T1: usize, const A2: usize, const T2: usize> {
    pub m1: M1,
    pub m2: M2,
}

impl<S: Zero + Scalar, M1, M2, const A1: usize, const T1: usize, const A2: usize, const T2: usize, const A_TOTAL: usize, const T_TOTAL: usize> 
    Manifold<S, A_TOTAL, T_TOTAL> for ProductSpace<M1, M2, A1, T1, A2, T2>
where
    // S: SubspaceScalar,
    M1: Manifold<S, A1, T1>,
    M2: Manifold<S, A2, T2>,
{
    type TangentVector = (M1::TangentVector, M2::TangentVector);

    #[inline]
    fn retract(&self, delta: &Self::TangentVector) -> Self {
        ProductSpace {
            m1: self.m1.retract(&delta.0),
            m2: self.m2.retract(&delta.1),
        }
    }

    #[inline]
    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        (
            self.m1.local_lift(&other.m1),
            self.m2.local_lift(&other.m2),
        )
    }

    fn pushforward_jacobian(&self) -> SMatrix<S, A_TOTAL, T_TOTAL> {
        let mut j_product = SMatrix::<S, A_TOTAL, T_TOTAL>::zeros();
        
        let j1 = self.m1.pushforward_jacobian();
        let j2 = self.m2.pushforward_jacobian();

        // Assign sub-matrices safely using nalgebra's fixed slice views on the stack.
        // These bounds are checked at runtime via assertions, which keeps the type checker happy.
        j_product.fixed_view_mut::<A1, T1>(0, 0).copy_from(&j1);
        j_product.fixed_view_mut::<A2, T2>(A1, T1).copy_from(&j2);

        j_product
    }

    #[inline]
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, A_TOTAL> {
        let mut out_ambient = SVector::<S, A_TOTAL>::zeros();

        // Compute the independent sub-manifold pushforwards matrix-free!
        let am1 = self.m1.apply_pushforward(&tangent.0);
        let am2 = self.m2.apply_pushforward(&tangent.1);

        // Splice the resulting small vectors directly into the stack output array
        out_ambient.fixed_rows_mut::<A1>(0).copy_from(&am1);
        out_ambient.fixed_rows_mut::<A2>(A1).copy_from(&am2);

        out_ambient
    }

    #[inline]
    fn vector_to_tangent(vec: &SVector<S, T_TOTAL>) -> Self::TangentVector {
        let v1 = vec.fixed_rows::<T1>(0).into_owned();
        let v2 = vec.fixed_rows::<T2>(T1).into_owned();
        (M1::vector_to_tangent(&v1), M2::vector_to_tangent(&v2))
    }

    #[inline]
    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, T_TOTAL> {
        let mut vec = SVector::<S, T_TOTAL>::zeros();
        vec.fixed_rows_mut::<T1>(0).copy_from(&M1::tangent_to_vector(&tangent.0));
        vec.fixed_rows_mut::<T2>(T1).copy_from(&M2::tangent_to_vector(&tangent.1));
        vec
    }
}




#[derive(Clone, Debug)]
pub struct ComplexRotation {
    // Ambient storage: x and y components of a unit complex number
    pub z: SVector<f64, 2>, 
}

impl ComplexRotation {
    pub fn identity() -> Self {
        Self { z: SVector::<f64, 2>::new(1.0, 0.0) }
    }
}

impl Manifold<f64, 2, 1> for ComplexRotation {
    type TangentVector = f64; // Local 1D angular perturbation dtheta

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        // Multiplicative exponential mapping: z_next = z * exp(i * dtheta)
        let cos_d = delta.cos();
        let sin_d = delta.sin();
        let x_next = self.z[0] * cos_d - self.z[1] * sin_d;
        let y_next = self.z[1] * cos_d + self.z[0] * sin_d;
        Self { z: SVector::<f64, 2>::new(x_next, y_next) }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        // dtheta = atan2(z2 x z1^*)
        let x_diff = other.z[0] * self.z[0] + other.z[1] * self.z[1];
        let y_diff = other.z[1] * self.z[0] - other.z[0] * self.z[1];
        y_diff.atan2(x_diff)
    }

    fn pushforward_jacobian(&self) -> SMatrix<f64, 2, 1> {
        // J_oplus = [-y, x]^T
        SMatrix::<f64, 2, 1>::new(-self.z[1], self.z[0])
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<f64, 2> {
        // Matrix-free: J_oplus * dtheta = [-y * dtheta, x * dtheta]^T
        SVector::<f64, 2>::new(-self.z[1] * tangent, self.z[0] * tangent)
    }

    fn vector_to_tangent(vec: &SVector<f64, 1>) -> Self::TangentVector {
        vec[0]
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }
}

impl InvariantLieGroup<f64, 2, 1> for ComplexRotation {
    fn exp(omega: &Self::TangentVector) -> Self {
        Self { z: SVector::<f64, 2>::new(omega.cos(), omega.sin()) }
    }

    fn log(&self) -> Self::TangentVector {
        self.z[1].atan2(self.z[0])
    }

    fn compose(&self, other: &Self) -> Self {
        let x = self.z[0] * other.z[0] - self.z[1] * other.z[1];
        let y = self.z[0] * other.z[1] + self.z[1] * other.z[0];
        Self { z: SVector::<f64, 2>::new(x, y) }
    }

    fn inverse(&self) -> Self {
        Self { z: SVector::<f64, 2>::new(self.z[0], -self.z[1]) }
    }

    fn adjoint(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::identity() // SO(2) is abelian, Ad_g = Identity
    }

    fn small_adjoint(_omega: &Self::TangentVector) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::zeros()    // ad_omega = 0
    }
}


#[derive(Clone, Debug)]
pub struct RealLine {
    pub x: SVector<f64, 1>,
}

impl Manifold<f64, 1, 1> for RealLine {
    type TangentVector = f64;

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        Self { x: SVector::<f64, 1>::new(self.x[0] + delta) }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        other.x[0] - self.x[0]
    }

    fn pushforward_jacobian(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::identity()
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }

    fn vector_to_tangent(vec: &SVector<f64, 1>) -> Self::TangentVector {
        vec[0]
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }
}
