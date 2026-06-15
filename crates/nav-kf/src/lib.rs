use nalgebra::{SMatrix, SVector};

pub mod sim;

pub struct KalmanFilter<const N: usize, const M: usize> {
    pub x: SVector<f64, N>,    // state estimate
    pub p: SMatrix<f64, N, N>, // state covariance
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    /// Generate the intermediate prediction used to update the state
    pub fn predict(
        &mut self,
        f: &SMatrix<f64, N, N>, // state-transition
        q: &SMatrix<f64, N, N>, // process-noise
    ) {
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
    }

    pub fn update(
        &mut self,
        z: &SVector<f64, M>,    // measurement outcomes
        h: &SMatrix<f64, M, N>, // measurement matrix
        r: &SMatrix<f64, M, M>, // measurement noise
    ) -> (SVector<f64, M>, SMatrix<f64, M, M>) {
        // innovation
        let y = z - h * self.x;

        // innovation covariance
        let s = h * self.p * h.transpose() + r;

        // s must be invertible
        let s_inv = s
            .try_inverse()
            .expect("S (innovation covariance) is singular");

        //kalman-gain
        let k = self.p * h.transpose() * s_inv;

        //update state
        self.x += k * y;

        self.p = {
            let i = SMatrix::<f64, N, N>::identity();
            let ikh = i - k * h;
            ikh * self.p * ikh.transpose() + k * r * k.transpose()
            // (i - k * h) * self.p
        };

        // innovation and its covariance
        (y, s)
    }
}
