#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // Set the values of the matrices used by the Extended Kalman Filter
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

// Prediction function
void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

// Update function used by lydar
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

// Update function used by radar
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  // get values for conversion to polar coordinates
  float px = x_[0];
  float py = x_[1];
  float dx = x_[2];
  float dy = x_[3];

  // Set rho and phi given the px, py, dx, and dy values
  float rho = sqrt((px * px) + (py * py));
  float phi = atan(py / px);
  
  // Set a pi variable used by a couple functions. Not technically neccesary because most math libraries have pi.
  float pi = 3.141592653589793238462643;

  // Set boolean values used in the process of normalizing phi
  bool isLess = phi < -pi;
  bool isGreater = phi > pi;

  // Normalize phi to the range -pi to pi
  while(isLess || isGreater) {
    if(isLess) {
      phi += (2 * pi);
    } else {
      phi -= (2 * pi);
    }
    isLess = phi < -pi;
    isGreater = phi > pi;
  }

  // To prevent division by zero, make sure rho, px, and py are no smaller than 1*10^-7
  float epsilon = 1e-7;
  if(rho < epsilon && rho > -epsilon) {
    rho = epsilon;
  }
  if(px < epsilon && px > -epsilon) {
    px = epsilon;
  }
  if(py < epsilon && py > -epsilon) {
    py = epsilon;
  }
  
  // Set rho dot
  float rho_dot = ((px * dx) + (py * dy)) / rho;
  
  // Get the input values of rho, phi, and rho dot.
  float ipt_rho = z[0];
  float ipt_phi = z[1];
  float ipt_rho_dot = z[2];

  // To prevent division by zero, make sure the input rho is no smaller than 0.0001
  if(ipt_rho < 0.0001 && ipt_rho > -0.0001) {
    ipt_rho = 0.0001;
  }
  
  // Normalize the input phi similarly to the saved phi
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
  
  // When phi is at the boundary of pi and -pi, the Filter will not take into account that phi is an angle that is reset when it reaches -pi or pi and when the boundary is passed. This results in the new values of phi combining with the old ones to form near-zero values, which can be diastrous. To prevent this problem, I made it so if the value of the new phi is more than pi/2 different than the saved phi, it will set the saved phi to the new phi. I got part of the idea for this from https://github.com/gardenermike/extended-kalman-filter/blob/master/src/kalman_filter.cpp
  if((phi - ipt_phi) > pi/2 || (phi - ipt_phi) < -pi/2) {
    phi = ipt_phi;
  }
  
  // Initialize Vectors to hold the new rho, phi, and rho dot values.
  Eigen::VectorXd x_polar = VectorXd(3);
  Eigen::VectorXd z_corrected = VectorXd(3);
  
  // Put the rho, phi and rho dot values into the vectors.  
  x_polar << rho, phi, rho_dot;
  z_corrected << ipt_rho, ipt_phi, ipt_rho_dot;
  
  // Now do the actual Extended Kalman Filter computations.
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
