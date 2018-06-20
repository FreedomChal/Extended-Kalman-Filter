#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_[0];
  float py = x_[1];
  float dx = x_[2];
  float dy = x_[3];

  float rho = sqrt((px * px) + (py * py));
  float phi = atan(py / px);
  
  float pi = 3.141592653589793238462643;

  bool isLess = phi < -pi;
  bool isGreater = phi > pi;

  while(isLess || isGreater) {
    if(isLess) {
      phi += (2 * pi);
    } else {
      phi -= (2 * pi);
    }
    isLess = phi < -pi;
    isGreater = phi > pi;
  }

  if(rho < 0.0001 && rho > -0.0001) { // to prevent division by zero
    rho = 0.0001;
  }
  //if(px < 0.0001 && px > -0.0001) { // to prevent division by zero
  //  px = 0.0001;
  //}
  //if(py < 0.0001 && py > -0.0001) { // to prevent division by zero
  //  py = 0.0001;
  //}
  //if(dx < 0.0001 && dx > -0.0001) { // to prevent division by zero
  //  dx = 0.0001;
  //}
  //if(dy < 0.0001 && dy > -0.0001) { // to prevent division by zero
  //  dy = 0.0001;
  //}

  float rho_dot = ((px * dx) + (py * dy)) / rho;
  
  float ipt_rho = z[0];
  float ipt_phi = z[1];
  float ipt_rho_dot = z[2];

  //if(ipt_rho < 0.0001 && ipt_rho > -0.0001) { // to prevent division by zero
  //  ipt_rho = 0.0001;
  //}
  
  isLess = ipt_phi < -pi;
  isGreater = ipt_phi > pi;

  while(isLess || isGreater) {
    if(isLess) {
      ipt_phi += (2 * pi);
    } else {
      ipt_phi -= (2 * pi);
    }
    isLess = ipt_phi < -pi;
    isGreater = ipt_phi > pi;
  }
  
  //if((phi - ipt_phi) > pi/2 || (phi - ipt_phi) < -pi/2) {
  //  ipt_phi += pi;
  //} // remove the issues caused with averaging values at the boundary of -pi and pi

  Eigen::VectorXd x_polar = VectorXd(3);
  Eigen::VectorXd z_corrected = VectorXd(3);
  
  x_polar << rho, phi, rho_dot;
  z_corrected << ipt_rho, ipt_phi, ipt_rho_dot;

  VectorXd y = z_corrected - x_polar;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
